#include "OperatorDescriptor.h"

#include <iostream>
#include <fstream>
#include <algorithm> 
#include <cctype>
#include <assert.h>

#include "DataTypeDescriptor.h"

#include "../utils/StringUtils.h"

#include "../json/JsonArray.h"
#include "../json/JsonObject.h"
#include "../json/JsonValue.h"

OperatorDescriptor::OperatorDescriptor(const std::string &fileName, const std::string &nameWithUS)
{
    this->m_FileName = fileName;
    this->m_NameWithUS = nameWithUS;
}

OperatorDescriptor::~OperatorDescriptor()
{
}

//gets string up to a first space, the rest is saved into line
std::string get_line_info(std::string &line)
{
    std::string info = "";
    for (auto iter = line.begin(); iter <= line.end(); iter++)
    {
        if (iter == line.end() || *iter == ' ')
        {
            if (iter != line.end())
            {
                line.erase(line.begin(), iter + 1);
            }
            break;
        }
        info += *iter;
    }
    return info;
}

std::vector<VariableDataType> parse_data_types(const std::string &line)
{
    std::vector<VariableDataType> data_types;
    std::string cur_data_type;
    for (auto iter = line.begin(); iter <= line.end(); iter++)
    {
        if (iter == line.end() || *iter == ' ')
        {
            data_types.push_back(GetDataTypeFromString(cur_data_type));
            cur_data_type = "";
            if (iter == line.end())
            {
                break;
            }
            continue;
        }
        cur_data_type += *iter;
    }
    return data_types;
}

std::vector<std::string> parse_parallel_info(const std::string &line)
{
    std::string cur_option;
    std::vector<std::string> fields;
    for (auto iter = line.begin(); iter <= line.end(); iter++)
    {
        if (iter == line.end() || *iter == ' ')
        {
            fields.push_back(cur_option);   
            cur_option = "";
            if (iter == line.end())
            {
                break;
            }
            continue;
        }

        cur_option += *iter;
    }

    assert(fields.size() >= 1);
    return fields;
}

void OperatorDescriptor::ParseFile()
{
    std::ifstream file;
    std::string line;
    int code_type = -1; // 0 - header, 1 - precond, 2 - op code, 3 - postcond
    std::string cur_op_name = "";

    std::string cur_code_cpu = "";
    std::string cur_code_npu = "";

    file.open(this->m_FileName);
    while (std::getline(file, line))
    {
        std::string trimmed = trim_copy(line);
        if (trimmed == "__head__")
        {
            code_type = 0;
            continue;
        }
        else if (trimmed == "__before__")
        {
            code_type = 1;
            continue;
        }
        else if (trimmed == "__operator__")
        {
            suboperator_t suboper;
            suboper.name = m_OpName;
            m_Suboperators.push_back(suboper);
            m_OperatorIndex = m_Suboperators.size() - 1;
            code_type = 20;
            continue;
        }
        else if (trimmed.find("__operator__ ") != std::string::npos)
        {
            cur_op_name = trimmed.substr(13);
            suboperator_t suboper;
            suboper.name = cur_op_name;
            m_Suboperators.push_back(suboper);
            code_type = 21;
            continue;
        }
        else if (trimmed == "__start__" && code_type == 21)
        {
            code_type = 22;
            continue;
        }
        else if (trimmed == "__after__")
        {
            code_type = 3;
            continue;
        }
        else if (trimmed == "__end__")
        {
            if (code_type != 0 && code_type != 1 && code_type != 3)
            {
                this->m_CodeFragment.push_back(cur_code_npu);
                this->m_CodeFragmentCPU.push_back(cur_code_cpu);
            }
            cur_code_cpu = "";
            cur_code_npu = "";
            code_type = -1;
            continue;
        }
        
        if (code_type >= 0)
        {
            if (code_type == 0)
            {
                this->m_HeaderCode += line;
                this->m_HeaderCode += "\n";
                continue;
            }
            else if (code_type == 1)
            {
                this->m_PrecondCode += line;
                this->m_PrecondCode += "\n";
                continue;
            }
            else if (code_type == 20)
            {
                cur_code_npu += line;
                cur_code_npu += "\n";

                cur_code_cpu += line;
                cur_code_cpu += "\n";
                continue;
            }
            else if (code_type == 22)
            {
                cur_code_cpu += line;
                cur_code_cpu += "\n";
                continue;
            }
            else if (code_type == 3)
            {
                this->m_PostcondCode += line;
                this->m_PostcondCode += "\n";
                continue;
            }
        }

        std::string info = get_line_info(trimmed);
        if (info == "name")
        {
            this->m_OpName = trimmed;
        }
        else if (info == "cfname")
        {
            this->m_CFName = trimmed;
        }
        else if (info == "types")
        {
            std::string info1 = get_line_info(trimmed);
            m_TypesInfo[info1] = parse_data_types(trimmed);
        }
        else if (info == "localvar")
        {
            VariableDescriptor varDesc;
            varDesc.ParseDescription(trimmed, m_TypesInfo, VariableDescriptor::VariableUsageType::LOCAL);
            this->m_VarDescriptors.push_back(varDesc);
        }
        else if (info == "opvar")
        {
            VariableDescriptor varDesc;
            varDesc.ParseDescription(trimmed, m_TypesInfo, VariableDescriptor::VariableUsageType::OPERATOR);
            this->m_VarDescriptors.push_back(varDesc);
            this->m_OperatorVarDescriptors.push_back(varDesc);
        }
        else if (info == "var" && code_type == 21)
        {
            subvar_t subvar;
            std::vector<std::string> fields = VariableDescriptor::ParseDescriptionRaw(trimmed);
            subvar.io_type = fields[0] == "input" ? VariableDescriptor::VariableIOType::INPUT : VariableDescriptor::VariableIOType::OUTPUT;
            subvar.name = fields[1];
            std::string dimType = fields[2];
            // assert(dimType == "explicit" || dimType == "implicit");
            if (dimType == "explicit")
            {
                subvar.dim_type = VariableDescriptor::VariableDimType::EXPLICIT;
                subvar.dimensions = VariableDescriptor::parse_variable_dimensions(fields[3]);
                
            }
            else
            {
                subvar.dim_type = VariableDescriptor::VariableDimType::IMPLICIT;
                subvar.dimensions_implicit = VariableDescriptor::parse_variable_dimensions_implicit(fields[3]);
            }
            m_Suboperators[m_Suboperators.size() - 1].vars.push_back(subvar);
        }
        else if (info == "attr")
        {
            AttributeDescriptor attrDesc;
            attrDesc.ParseDescription(trimmed);
            if (code_type == 21)
            {
                m_Suboperators[m_Suboperators.size() - 1].attrs.push_back(attrDesc);
            }
            else
            {
                this->m_AttrDescriptors.push_back(attrDesc);
            }
        }
        else if (info == "rule")
        {
            RuleDescriptor ruleDesc;
            ruleDesc.ParseDescription(trimmed);
            if (code_type == 21)
            {
                m_Suboperators[m_Suboperators.size() - 1].rules.push_back(ruleDesc);
            }
            else
            {
                this->m_RuleDescriptors.push_back(ruleDesc);
            }
        }
        else if (info == "parallel")
        {
            std::vector<std::string> fields = parse_parallel_info(trimmed);
            if (fields[0] == "true" || fields[0] == "1")
            {
                m_IsParallel = true;
            }
            for (int i = 1; i < fields.size(); i++)
            {
                if (fields[i].rfind("id=", 0) == 0)
                {
                    m_BlockIdName = fields[i].substr(3);
                    continue;
                }
                if (fields[i].rfind("dim=", 0) == 0)
                {
                    m_BlockDimName = fields[i].substr(4);
                    continue;
                }
            }
        }
    }
    file.close();
}

const std::string &OperatorDescriptor::GetFileName() const
{
    return this->m_FileName;
}

const std::string &OperatorDescriptor::GetOpName() const
{
    return this->m_OpName;
}

const std::string &OperatorDescriptor::GetCFName() const
{
    return this->m_CFName;
}

const std::string &OperatorDescriptor::GetNameWithUS() const
{
    return this->m_NameWithUS;
}

const std::vector<VariableDescriptor> &OperatorDescriptor::GetVarDescriptors() const
{
    return this->m_VarDescriptors;
}

const std::vector<VariableDescriptor> &OperatorDescriptor::GetOperatorVarDescriptors() const
{
    return this->m_OperatorVarDescriptors;
}

const std::vector<AttributeDescriptor> &OperatorDescriptor::GetAttrDescriptors() const
{
    return this->m_AttrDescriptors;
}

const std::vector<RuleDescriptor> &OperatorDescriptor::GetRuleDescriptors() const
{
    return this->m_RuleDescriptors;
}

const std::string &OperatorDescriptor::GetHeaderCode() const
{
    return this->m_HeaderCode;
}

const std::string &OperatorDescriptor::GetPrecondCode() const
{
    return this->m_PrecondCode;
}

const std::vector<std::string> &OperatorDescriptor::GetCodeFragment() const
{
    return this->m_CodeFragment;
}

const std::vector<std::string> &OperatorDescriptor::GetCodeFragmentCPU() const
{
    return this->m_CodeFragmentCPU;
}

const std::string &OperatorDescriptor::GetPostcondCode() const
{
    return this->m_PostcondCode;
}

bool OperatorDescriptor::IsParallel() const
{
    return m_IsParallel;
}

const std::string &OperatorDescriptor::GetBlockIdName() const
{
    return m_BlockIdName;
}

const std::string &OperatorDescriptor::GetBlockDimName() const
{
    return m_BlockDimName;
}

const std::vector<OperatorDescriptor::suboperator_t> &OperatorDescriptor::GetSuboperators() const
{
    return m_Suboperators;
}

int OperatorDescriptor::GetOperatorIndex() const
{
    return m_OperatorIndex;
}

bool OperatorDescriptor::ContainsRule(std::vector<VariableDescriptor> &vars, std::vector<int> comb) const
{
    for (auto rule : m_RuleDescriptors)
    {
        bool eq = true;
        int inIdx = 0, outIdx = 0;
        
        for (auto varDesc : vars)
        {
            if (varDesc.GetVariableIOType() == VariableDescriptor::VariableIOType::INPUT)
            {
                if (rule.GetInputDataType(inIdx) != varDesc.GetVariableDataTypes()[comb[inIdx + outIdx]])
                {
                    eq = false;
                    break;
                }
                inIdx++;
            }
            else
            {
                if (rule.GetOutputDataType(outIdx) != varDesc.GetVariableDataTypes()[comb[inIdx + outIdx]])
                {
                    eq = false;
                    break;
                }
                outIdx++;
            }
        }

        if (eq)
            return true;
    }

    return false;
}

json::JsonElement *OperatorDescriptor::CreateJson(std::vector<VariableDescriptor> &vars, std::vector<int> comb, int sub_idx) const
{
    //json::JsonArray *outerArray = new json::JsonArray("");
    json::JsonObject *outerObject = new json::JsonObject("");

    auto suboper = m_Suboperators[sub_idx];

    json::JsonValue *opName = new json::JsonValue("op", suboper.name);
    outerObject->AddField(opName);

    json::JsonArray *inputVars = new json::JsonArray("input_desc");
    json::JsonArray *outputVars = new json::JsonArray("output_desc");

    std::vector<VariableDataType> cur_types;

    if (sub_idx != m_OperatorIndex)
    {
        for (auto iter2 = suboper.vars.begin(); iter2 < suboper.vars.end(); iter2++)
        {
            int idx = 0;
            for (auto iter = m_VarDescriptors.begin(); iter < m_VarDescriptors.end(); iter++)
            {
                if (iter->GetVariableName() == iter2->name)
                {
                    cur_types.push_back(iter->GetVariableDataTypes()[comb[idx]]);
                    break;
                }
                idx++;
            }
        }
    }

    int idx = 0;
    for (auto iter = m_VarDescriptors.begin(); iter < m_VarDescriptors.end(); iter++)
    {
        if (iter->GetVariableUsageType() != VariableDescriptor::VariableUsageType::OPERATOR)
        {
            idx++;
            continue;
        }

        json::JsonObject *varDesc = new json::JsonObject("");

        json::JsonValue *varFormat = new json::JsonValue("format", "ND");
        json::JsonArray *varDims = new json::JsonArray("shape");
        json::JsonArray *varRanges = new json::JsonArray("shape_range");

        VariableDataType need_type = iter->GetVariableDataTypes()[comb[idx]];
        int index2 = 0;

        if (sub_idx != m_OperatorIndex)
        {
            bool found_name = false;

            for (auto iter2 = suboper.vars.begin(); iter2 < suboper.vars.end(); iter2++)
            {
                if (iter->GetVariableName() == iter2->name)
                {
                    found_name = true;
                    break;
                }
                index2++;
            }

            if (!found_name)
            {
                continue;
            }

            for (int i = 0; i < suboper.rules.size(); i++)
            {
                bool found = true;
                for (int j = 0; j < cur_types.size(); j++)
                {
                    if (cur_types[j] != suboper.rules[i].GetInputDataType(j))
                    {
                        found = false;
                        break;
                    }
                }
                if (found)
                {
                     need_type = suboper.rules[i].GetOutputDataType(index2);
                     break;
                }
            }  
        }

        json::JsonValue *varType = new json::JsonValue("type", GetJsonTypeFromDataType(need_type));

        int numNegatives = 0;

        if (sub_idx != m_OperatorIndex)
        {
            if (suboper.vars[index2].dim_type == VariableDescriptor::VariableDimType::EXPLICIT) 
            {
                auto dims = suboper.vars[index2].dimensions;
                for (auto iter1 = dims.begin(); iter1 < dims.end(); iter1++)
                {
                    int val = *iter1;
                    if (val < 0)
                    {
                        numNegatives++;

                        json::JsonArray *range = new json::JsonArray("");
                        json::JsonValue *rangeBegin = new json::JsonValue("", 1);
                        json::JsonValue *rangeEnd = new json::JsonValue("", -1);
                        range->AddElement(rangeBegin);
                        range->AddElement(rangeEnd);

                        varRanges->AddElement(range);
                    }

                    json::JsonValue *varDimVal = new json::JsonValue("", val);
                    varDims->AddElement(varDimVal);
                }
            }
            else
            {
                auto dims = suboper.vars[index2].dimensions_implicit;
                for (auto iter1 = dims.begin(); iter1 < dims.end(); iter1++)
                {
                    int val = -1; //*iter1;
                    if (val < 0)
                    {
                        numNegatives++;

                        json::JsonArray *range = new json::JsonArray("");
                        json::JsonValue *rangeBegin = new json::JsonValue("", 1);
                        json::JsonValue *rangeEnd = new json::JsonValue("", -1);
                        range->AddElement(rangeBegin);
                        range->AddElement(rangeEnd);

                        varRanges->AddElement(range);
                    }

                    json::JsonValue *varDimVal = new json::JsonValue("", val);
                    varDims->AddElement(varDimVal);
                }
            }
        }
        else
        {
            if (iter->GetVariableIOType() == VariableDescriptor::VariableIOType::INPUT || 
                (iter->GetVariableIOType() == VariableDescriptor::VariableIOType::OUTPUT && 
                iter->GetVariableDimType() == VariableDescriptor::VariableDimType::EXPLICIT)) {
                
                auto dims = iter->GetVariableDimensions();
                for (auto iter1 = dims.begin(); iter1 < dims.end(); iter1++)
                {
                    int val = *iter1;
                    if (val < 0)
                    {
                        numNegatives++;

                        json::JsonArray *range = new json::JsonArray("");
                        json::JsonValue *rangeBegin = new json::JsonValue("", 1);
                        json::JsonValue *rangeEnd = new json::JsonValue("", -1);
                        range->AddElement(rangeBegin);
                        range->AddElement(rangeEnd);

                        varRanges->AddElement(range);
                    }

                    json::JsonValue *varDimVal = new json::JsonValue("", val);
                    varDims->AddElement(varDimVal);
                }
            }
            else
            {
                auto dims = iter->GetVariableDimensionsImplicit();
                for (auto iter1 = dims.begin(); iter1 < dims.end(); iter1++)
                {
                    int val = -1; //*iter1;
                    if (val < 0)
                    {
                        numNegatives++;

                        json::JsonArray *range = new json::JsonArray("");
                        json::JsonValue *rangeBegin = new json::JsonValue("", 1);
                        json::JsonValue *rangeEnd = new json::JsonValue("", -1);
                        range->AddElement(rangeBegin);
                        range->AddElement(rangeEnd);

                        varRanges->AddElement(range);
                    }

                    json::JsonValue *varDimVal = new json::JsonValue("", val);
                    varDims->AddElement(varDimVal);
                }
            }
        }
        
        varDesc->AddField(varFormat);
        varDesc->AddField(varDims);
        if (numNegatives > 0) 
        {
            varDesc->AddField(varRanges);
        }
        varDesc->AddField(varType);        

        if (iter->GetVariableIOType() == VariableDescriptor::VariableIOType::INPUT)
        {
            inputVars->AddElement(varDesc);
        }
        else 
        {
            outputVars->AddElement(varDesc);
        }
        idx++;
    }

    json::JsonArray *attrs = new json::JsonArray("attr");

    std::vector<AttributeDescriptor> attrsDesc;

    if (sub_idx != m_OperatorIndex)
    {
        attrsDesc = suboper.attrs;
    }
    else
    {
        attrsDesc = m_AttrDescriptors;
    }

    for (auto iter = attrsDesc.begin(); iter < attrsDesc.end(); iter++)
    {
        json::JsonObject *attrDesc = new json::JsonObject("");

        json::JsonValue *attrName = new json::JsonValue("name", iter->GetAttributeName());
        json::JsonValue *attrType = new json::JsonValue("type", GetJsonTypeFromDataType(iter->GetAttributeDataType()));
        json::JsonValue *attrValue = new json::JsonValue("value", iter->GetAttributeValue(), false);

        attrDesc->AddField(attrName);
        attrDesc->AddField(attrType);
        attrDesc->AddField(attrValue);

        attrs->AddElement(attrDesc);
    }

    outerObject->AddField(inputVars);
    outerObject->AddField(outputVars);
    outerObject->AddField(attrs);

    //outerArray->AddElement(outerObject);
    return outerObject;
}


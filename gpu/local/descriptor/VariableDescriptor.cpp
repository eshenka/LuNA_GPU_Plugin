#include "VariableDescriptor.h"

#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <string>

//  constructor
VariableDescriptor::VariableDescriptor()
{
}

//  deconstructor
VariableDescriptor::~VariableDescriptor()
{
}

VariableDescriptor::VariableIOType parse_variable_type(const std::string varTypeStr)
{
    if (varTypeStr == "input" || varTypeStr == "Input" || varTypeStr == "INPUT")
    {
        return VariableDescriptor::VariableIOType::INPUT;
    }
    else if (varTypeStr == "output" || varTypeStr == "Output" || varTypeStr == "OUTPUT")
    {
        return VariableDescriptor::VariableIOType::OUTPUT;
    }
    else 
    {
        throw std::invalid_argument("Can't parse variable type. Should be INPUT or OUTPUT.");
    }
}

std::vector<int> VariableDescriptor::parse_variable_dimensions(const std::string varDimsStr)
{
    std::vector<int> dims;
    std::string curDim;

    for (auto iter = varDimsStr.begin(); iter <= varDimsStr.end(); iter++)
    {
        if (iter == varDimsStr.end() || *iter == ',')
        {
            dims.push_back(std::stoi(curDim));
            curDim = "";
            if (iter == varDimsStr.end())
            {
                break;
            }
            continue;
        }

        if (*iter == ' ')
        {
            continue;
        }

        curDim += *iter;
    }

    return dims;
}

std::vector<std::string> VariableDescriptor::parse_variable_dimensions_implicit(const std::string varDimsStr)
{
    std::vector<std::string> dims;
    std::string curDim;

    for (auto iter = varDimsStr.begin(); iter <= varDimsStr.end(); iter++)
    {
        if (iter == varDimsStr.end() || *iter == ',')
        {
            dims.push_back(curDim);
            curDim = "";
            if (iter == varDimsStr.end())
            {
                break;
            }
            continue;
        }

        if (*iter == ' ')
        {
            continue;
        }

        curDim += *iter;
    }

    return dims;
}

std::vector<std::string> VariableDescriptor::ParseDescriptionRaw(const std::string &description)
{
    std::vector<std::string> fields;
    std::string curField = "";
    int bracketBalance = 0;
    for (auto iter = description.begin(); iter <= description.end(); iter++)
    {
        if (iter == description.end() || (*iter == ' ' && bracketBalance == 0))
        {
            fields.push_back(curField);
            curField = "";
            if (iter == description.end())
            {
                break;
            }
            continue;
        }

        if (*iter == '[')
        {
            bracketBalance++;
            if (bracketBalance == 1)
            {
                continue;
            }
        }
        if (*iter == ']')
        {
            bracketBalance--;
            if (bracketBalance == 0)
            {
                continue;
            }
        }

        curField += *iter;
    }

    return fields;
}

//  parse line from description file
void VariableDescriptor::ParseDescription(const std::string &description, 
                                            const std::map<std::string, std::vector<VariableDataType>> &typesInfo,
                                            VariableUsageType usageType)
{
    std::vector<std::string> fields = ParseDescriptionRaw(description);

    m_VarUsageType = usageType;

    assert(fields.size() > 0);

    m_VarIOType = parse_variable_type(fields[0]);

    assert((fields.size() >= 4 && m_VarIOType == VariableIOType::INPUT) || 
            (fields.size() >= 5 && m_VarIOType == VariableIOType::OUTPUT));

    auto iter = typesInfo.find(fields[1]);
    if (iter == typesInfo.end()) {
        throw new std::invalid_argument("Can't parse variable data types");
    }
    m_VarDataTypes = iter->second;
    m_VarName = fields[2];
    if (m_VarIOType == VariableIOType::INPUT)
    {
        m_VarDimensions = parse_variable_dimensions(fields[3]);

        m_VarSizeName = fields[4];
        //m_VarDimsName = fields[5];
    }
    else
    {
        std::string dimType = fields[3];
        assert(dimType == "explicit" || dimType == "implicit");
        if (dimType == "explicit")
        {
            m_VarDimType = VariableDimType::EXPLICIT;
            m_VarDimensions = parse_variable_dimensions(fields[4]);
            
        }
        else
        {
            m_VarDimType = VariableDimType::IMPLICIT;
            m_VarDimensionsImplicit = parse_variable_dimensions_implicit(fields[4]);
        }
        m_VarSizeName = fields[5];
    }

    m_VarSizeName = std::string("size") + m_VarName;
    m_VarTypeName = std::string("Type") + m_VarName;
    m_VarIndexName = std::string("IDX_") + m_VarName;

    for (int i = 4; i < fields.size(); i++)
    {
        if (fields[i].rfind("size=", 0) == 0)
        {
            m_VarSizeName = fields[i].substr(5);
            continue;
        }
        if (fields[i].rfind("type=", 0) == 0)
        {
            m_VarTypeName = fields[i].substr(5);
            continue;
        }
        if (fields[i].rfind("indx=", 0) == 0)
        {
            m_VarIndexName = fields[i].substr(5);
            continue;
        }
    }
}

VariableDescriptor::VariableIOType VariableDescriptor::GetVariableIOType() const
{
    return m_VarIOType;
}

VariableDescriptor::VariableUsageType VariableDescriptor::GetVariableUsageType() const
{
    return m_VarUsageType;
}

VariableDescriptor::VariableDimType VariableDescriptor::GetVariableDimType() const
{
    return m_VarDimType;
}

const std::vector<VariableDataType> &VariableDescriptor::GetVariableDataTypes() const
{
    return m_VarDataTypes;
}

const std::string &VariableDescriptor::GetVariableName() const
{
    return m_VarName;
}

const std::vector<int> &VariableDescriptor::GetVariableDimensions() const
{
    return m_VarDimensions;
}

const std::vector<std::string> &VariableDescriptor::GetVariableDimensionsImplicit() const
{
    return m_VarDimensionsImplicit;
}

const std::string &VariableDescriptor::GetVariableSizeName() const
{
    return m_VarSizeName;
}

const std::string &VariableDescriptor::GetVariableTypeName() const
{
    return m_VarTypeName;
}

const std::string &VariableDescriptor::GetVariableIndexName() const
{
    return m_VarIndexName;
}

int64_t VariableDescriptor::GetNumDimensions() const
{
    if (m_VarIOType == INPUT)
        return m_VarDimensions.size();
    if (m_VarDimType == EXPLICIT)
        return m_VarDimensions.size();
    return m_VarDimensionsImplicit.size();
}

#pragma once

#include <string>
#include <vector>
#include <map>

#include "VariableDescriptor.h"
#include "AttributeDescriptor.h"
#include "RuleDescriptor.h"
#include "../json/JsonElement.h"


class OperatorDescriptor
{
    typedef struct subvar_t
    {
        VariableDescriptor::VariableIOType io_type;
        VariableDescriptor::VariableDimType dim_type;
        std::string name;
        std::vector<int> dimensions;
        std::vector<std::string> dimensions_implicit;
    } subvar_t;

    typedef struct suboperator_t
    {
        std::string name;
        std::vector<subvar_t> vars;
        std::vector<RuleDescriptor> rules;
        std::vector<AttributeDescriptor> attrs;
    } suboperator_t;

private:
    std::string m_FileName;

    std::string m_OpName;

    std::string m_CFName;

    std::string m_NameWithUS;

    std::vector<VariableDescriptor> m_VarDescriptors;

    std::vector<VariableDescriptor> m_OperatorVarDescriptors;

    std::vector<AttributeDescriptor> m_AttrDescriptors;

    std::vector<RuleDescriptor> m_RuleDescriptors;

    std::string m_HeaderCode = "";

    std::string m_PrecondCode = "";

    std::vector<std::string> m_CodeFragment;

    std::vector<std::string> m_CodeFragmentCPU;

    std::string m_PostcondCode = "";
    
    std::map<std::string, std::vector<VariableDataType>> m_TypesInfo;

    bool m_IsParallel = false;

    std::string m_BlockIdName = "blockid";
    
    std::string m_BlockDimName = "blockdim";

    std::vector<suboperator_t> m_Suboperators;

    int m_OperatorIndex = -1;

public:
    //  constructor
    OperatorDescriptor(const std::string &fileName, const std::string &nameWithUS);

    //  destructor
    ~OperatorDescriptor();

    //  read and parse file
    void ParseFile();

    const std::string &GetFileName() const;

    const std::string &GetOpName() const;

    const std::string &GetCFName() const;

    const std::string &GetNameWithUS() const;

    const std::vector<VariableDescriptor> &GetVarDescriptors() const;

    const std::vector<VariableDescriptor> &GetOperatorVarDescriptors() const;

    const std::vector<AttributeDescriptor> &GetAttrDescriptors() const;

    const std::vector<RuleDescriptor> &GetRuleDescriptors() const;

    const std::string &GetHeaderCode() const;

    const std::string &GetPrecondCode() const;

    const std::vector<std::string> &GetCodeFragment() const;

    const std::vector<std::string> &GetCodeFragmentCPU() const;

    const std::string &GetPostcondCode() const;

    bool IsParallel() const;

    const std::string &GetBlockIdName() const;

    const std::string &GetBlockDimName() const;

    const std::vector<suboperator_t> &GetSuboperators() const;

    int GetOperatorIndex() const;

    bool ContainsRule(std::vector<VariableDescriptor> &vars, std::vector<int> comb) const;

    json::JsonElement *CreateJson(std::vector<VariableDescriptor> &vars, std::vector<int> comb, int sub_idx) const;
};

#include "RuleDescriptor.h"

void RuleDescriptor::SetInputDataTypes(const std::vector<VariableDataType> &inputTypes)
{
    m_InputTypes = inputTypes;
}

void RuleDescriptor::SetOutputDataTypes(const std::vector<VariableDataType> &outputTypes)
{
    m_OutputTypes = outputTypes;
}

void RuleDescriptor::ParseDescription(const std::string &description)
{
    std::string curField = "";
    bool isOutput = false;
    for (auto iter = description.begin(); iter <= description.end(); iter++)
    {
        if (iter == description.end() || *iter == ' ')
        {
            if (curField == ">")
            {
                isOutput = true;
                curField = "";
                continue;
            }

            VariableDataType type = GetDataTypeFromString(curField);
            if (isOutput)
                m_OutputTypes.push_back(type);
            else
                m_InputTypes.push_back(type);
            
            curField = "";
            if (iter == description.end())
            {
                break;
            }
            continue;
        }

        curField += *iter;
    }
}

VariableDataType RuleDescriptor::GetInputDataType(int index)
{
    return m_InputTypes[index];
}

VariableDataType RuleDescriptor::GetOutputDataType(int index)
{
    return m_OutputTypes[index];
}

const std::vector<VariableDataType> &RuleDescriptor::GetOutputDataTypes() const
{
    return m_OutputTypes;
}


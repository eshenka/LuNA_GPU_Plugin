#include <vector>
#include "DataTypeDescriptor.h"

class RuleDescriptor
{
public:
    RuleDescriptor() = default;
    ~RuleDescriptor() = default;

    void SetInputDataTypes(const std::vector<VariableDataType> &inputTypes);
    void SetOutputDataTypes(const std::vector<VariableDataType> &outputTypes);

    VariableDataType GetInputDataType(int index);
    VariableDataType GetOutputDataType(int index);
    const std::vector<VariableDataType> &GetOutputDataTypes() const;

    void ParseDescription(const std::string &description);

private:

    std::vector<VariableDataType> m_InputTypes;
    std::vector<VariableDataType> m_OutputTypes;

};

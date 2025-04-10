#pragma once

#include <string>
#include <vector>
#include <map>

#include "DataTypeDescriptor.h"

class VariableDescriptor
{
public: //enum variable input/output type
    typedef enum {
        INPUT,
        OUTPUT
    } VariableIOType;

    typedef enum {
        OPERATOR,
        LOCAL
    } VariableUsageType;

    typedef enum {
        EXPLICIT,
        IMPLICIT
    } VariableDimType;

private:
    //  type of the variable (input/output)
    VariableIOType m_VarIOType;

    VariableUsageType m_VarUsageType;

    //  type of dimensions (explicit/implicit)
    VariableDimType m_VarDimType;

    //  variable name
    std::string m_VarName;

    //  dimensions (sizes) of variable as Tensor
    std::vector<int> m_VarDimensions;

    //  dimensions (sizes) of variable as Tensor
    std::vector<std::string> m_VarDimensionsImplicit;

    //  name of the variable that will contain current dimensions
    std::string m_VarSizeName;

    //  name of the variable that will contain current type
    std::string m_VarTypeName;

        //  name of the variable that will provide indexing
    std::string m_VarIndexName;

    //  list of types that this variable supports
    std::vector<VariableDataType> m_VarDataTypes;

public:

    //  constructor
    VariableDescriptor();

    //  deconstructor
    ~VariableDescriptor();

    //  parse line from description file
    void ParseDescription(const std::string &description, 
                            const std::map<std::string, std::vector<VariableDataType>> &typesInfo,
                            VariableUsageType usageType);

    static std::vector<std::string> ParseDescriptionRaw(const std::string &description);

    static std::vector<int> parse_variable_dimensions(const std::string varDimsStr);

    static std::vector<std::string> parse_variable_dimensions_implicit(const std::string varDimsStr);

public: //  getters

    VariableIOType GetVariableIOType() const;

    VariableUsageType GetVariableUsageType() const;

    VariableDimType GetVariableDimType() const;

    const std::vector<VariableDataType> &GetVariableDataTypes() const;

    const std::string &GetVariableName() const;

    const std::vector<int> &GetVariableDimensions() const;

    const std::vector<std::string> &GetVariableDimensionsImplicit() const;

    const std::string &GetVariableSizeName() const;

    const std::string &GetVariableTypeName() const;

    const std::string &GetVariableIndexName() const;

    int64_t GetNumDimensions() const;
};


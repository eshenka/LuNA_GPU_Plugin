#pragma once

#include <string>

//enum variable data type
typedef enum {
    FLOAT16,
    FLOAT,
    DOUBLE,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    BOOL,
    COMPLEX64,
    COMPLEX128,
    OTHER
} VariableDataType;

VariableDataType GetDataTypeFromString(const std::string &str);

const std::string GetCPPTypeFromDataType(VariableDataType dataType);

const std::string GetTensorTypeFromDataType(VariableDataType dataType);

const std::string GetACLTypeFromDataType(VariableDataType dataType);

const std::string GetShortNameFromDataType(VariableDataType dataType);

const std::string GetJsonTypeFromDataType(VariableDataType dataType);

const std::string GetAttrFunctionNameFromDataType(VariableDataType dataType);


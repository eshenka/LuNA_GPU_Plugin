#include "DataTypeDescriptor.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

VariableDataType GetDataTypeFromString(const std::string &str)
{
    std::string dataTypeStr(str);
    std::transform(dataTypeStr.begin(), dataTypeStr.end(), dataTypeStr.begin(),
                     [](unsigned char c){ return std::tolower(c); });

    if (dataTypeStr == "float16" || dataTypeStr == "half")
    {
        return VariableDataType::FLOAT16;
    }
    if (dataTypeStr == "float" || dataTypeStr == "single")
    {
        return VariableDataType::FLOAT;
    }
    if (dataTypeStr == "double")
    {
        return VariableDataType::DOUBLE;
    }
    if (dataTypeStr == "char" || dataTypeStr == "int8")
    {
        return VariableDataType::INT8;
    }
    if (dataTypeStr == "short" || dataTypeStr == "int16")
    {
        return VariableDataType::INT16;
    }
    if (dataTypeStr == "int" || dataTypeStr == "int32")
    {
        return VariableDataType::INT32;
    }
    if (dataTypeStr == "long" || dataTypeStr == "int64")
    {
        return VariableDataType::INT64;
    }
    if (dataTypeStr == "uint8")
    {
        return VariableDataType::UINT8;
    }
    if (dataTypeStr == "uint16")
    {
        return VariableDataType::UINT16;
    }
    if (dataTypeStr == "uint32")
    {
        return VariableDataType::UINT32;
    }
    if (dataTypeStr == "uint64")
    {
        return VariableDataType::UINT64;
    }
    if (dataTypeStr == "bool" || dataTypeStr == "boolean")
    {
        return VariableDataType::BOOL;
    }
    if (dataTypeStr == "complex64")
    {
        return VariableDataType::COMPLEX64;
    }
    if (dataTypeStr == "complex128")
    {
        return VariableDataType::COMPLEX128;
    }
}

const std::string GetCPPTypeFromDataType(VariableDataType dataType)
{
    switch(dataType)
    {
        case VariableDataType::FLOAT16:
            return "Eigen::half";
        case VariableDataType::FLOAT:
            return "float";
        case VariableDataType::DOUBLE:
            return "double";
        case VariableDataType::INT8:
            return "int8_t";
        case VariableDataType::INT16:
            return "int16_t";
        case VariableDataType::INT32:
            return "int32_t";
        case VariableDataType::INT64:
            return "int64_t";
        case VariableDataType::UINT8:
            return "uint8_t";
        case VariableDataType::UINT16:
            return "uint16_t";
        case VariableDataType::UINT32:
            return "uint32_t";
        case VariableDataType::UINT64:
            return "uint64_t";
        case VariableDataType::BOOL:
            return "bool";
        case VariableDataType::COMPLEX64:
            return "std::complex<float>";
        case VariableDataType::COMPLEX128:
            return "std::complex<double>";
    }
}

const std::string GetTensorTypeFromDataType(VariableDataType dataType)
{
    switch(dataType)
    {
        case VariableDataType::FLOAT16:
            return "DT_FLOAT16";
        case VariableDataType::FLOAT:
            return "DT_FLOAT";
        case VariableDataType::DOUBLE:
            return "DT_DOUBLE";
        case VariableDataType::INT8:
            return "DT_INT8";
        case VariableDataType::INT16:
            return "DT_INT16";
        case VariableDataType::INT32:
            return "DT_INT32";
        case VariableDataType::INT64:
            return "DT_INT64";
        case VariableDataType::UINT8:
            return "DT_UINT8";
        case VariableDataType::UINT16:
            return "DT_UINT16";
        case VariableDataType::UINT32:
            return "DT_UINT32";
        case VariableDataType::UINT64:
            return "DT_UINT64";
        case VariableDataType::BOOL:
            return "DT_BOOL";
        case VariableDataType::COMPLEX64:
            return "DT_COMPLEX64";
        case VariableDataType::COMPLEX128:
            return "DT_COMPLEX128";
    }
}

const std::string GetACLTypeFromDataType(VariableDataType dataType)
{
    switch(dataType)
    {
        case VariableDataType::FLOAT16:
            return "ACL_FLOAT16";
        case VariableDataType::FLOAT:
            return "ACL_FLOAT";
        case VariableDataType::DOUBLE:
            return "ACL_DOUBLE";
        case VariableDataType::INT8:
            return "ACL_INT8";
        case VariableDataType::INT16:
            return "ACL_INT16";
        case VariableDataType::INT32:
            return "ACL_INT32";
        case VariableDataType::INT64:
            return "ACL_INT64";
        case VariableDataType::UINT8:
            return "ACL_UINT8";
        case VariableDataType::UINT16:
            return "ACL_UINT16";
        case VariableDataType::UINT32:
            return "ACL_UINT32";
        case VariableDataType::UINT64:
            return "ACL_UINT64";
        case VariableDataType::BOOL:
            return "ACL_BOOL";
        case VariableDataType::COMPLEX64:
            return "ACL_COMPLEX64";
        case VariableDataType::COMPLEX128:
            return "ACL_COMPLEX128";
    }
}

const std::string GetShortNameFromDataType(VariableDataType dataType)
{
    switch(dataType)
    {
        case VariableDataType::FLOAT16:
            return "h";
        case VariableDataType::FLOAT:
            return "f";
        case VariableDataType::DOUBLE:
            return "d";
        case VariableDataType::INT8:
            return "i1";
        case VariableDataType::INT16:
            return "i2";
        case VariableDataType::INT32:
            return "i4";
        case VariableDataType::INT64:
            return "i8";
        case VariableDataType::UINT8:
            return "u1";
        case VariableDataType::UINT16:
            return "u2";
        case VariableDataType::UINT32:
            return "u4";
        case VariableDataType::UINT64:
            return "u8";
        case VariableDataType::BOOL:
            return "b";
        case VariableDataType::COMPLEX64:
            return "c8";
        case VariableDataType::COMPLEX128:
            return "c16";
    } 
}


const std::string GetJsonTypeFromDataType(VariableDataType dataType)
{
    switch(dataType)
    {
        case VariableDataType::FLOAT16:
            return "float16";
        case VariableDataType::FLOAT:
            return "float";
        case VariableDataType::DOUBLE:
            return "double";
        case VariableDataType::INT8:
            return "int8";
        case VariableDataType::INT16:
            return "int16";
        case VariableDataType::INT32:
            return "int32";
        case VariableDataType::INT64:
            return "int64";
        case VariableDataType::UINT8:
            return "uint8";
        case VariableDataType::UINT16:
            return "uint16";
        case VariableDataType::UINT32:
            return "uint32";
        case VariableDataType::UINT64:
            return "uint64";
        case VariableDataType::BOOL:
            return "bool";
        case VariableDataType::COMPLEX64:
            return "complex64";
        case VariableDataType::COMPLEX128:
            return "complex128";
    } 
}

const std::string GetAttrFunctionNameFromDataType(VariableDataType dataType)
{
    switch(dataType)
    {
        case VariableDataType::FLOAT:
            return "aclopSetAttrFloat";
        case VariableDataType::INT8:
        case VariableDataType::INT16:
        case VariableDataType::INT32:
        case VariableDataType::INT64:
            return "aclopSetAttrInt";
        case VariableDataType::BOOL:
            return "aclopSetAttrBool";
        default:
            throw std::invalid_argument("Attributes only support float, bool and signed ints (int8, int16, int32, int64)");
    } 
}


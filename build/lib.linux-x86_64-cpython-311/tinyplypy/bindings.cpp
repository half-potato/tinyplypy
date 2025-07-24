#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#include <fstream>
#include <sstream>
#include <memory>
#include <unordered_map>

//-----------------------------------------------------------------
// Helpers to map between NumPy dtypes and tinyply::Type
//-----------------------------------------------------------------
inline tinyply::Type numpy_dtype_to_tinyply_type(const pybind11::dtype &dt)
{
    // This is a minimal mapping. Extend as necessary for other types.
    // We match .kind() and .itemsize() for quick mapping.
    // E.g. for float32 => .kind() == 'f' and .itemsize() == 4
    const char kind = dt.kind();
    const size_t itemsize = dt.itemsize();

    if      (kind == 'f' && itemsize == 4)  return tinyply::Type::FLOAT32;
    // else if (kind == 'f' && itemsize == 8)  return tinyply::Type::FLOAT64;
    else if (kind == 'i' && itemsize == 1)  return tinyply::Type::INT8;
    else if (kind == 'u' && itemsize == 1)  return tinyply::Type::UINT8;
    else if (kind == 'i' && itemsize == 2)  return tinyply::Type::INT16;
    else if (kind == 'u' && itemsize == 2)  return tinyply::Type::UINT16;
    else if (kind == 'i' && itemsize == 4)  return tinyply::Type::INT32;
    else if (kind == 'u' && itemsize == 4)  return tinyply::Type::UINT32;
    /*else if (kind == 'i' && itemsize == 8)  return tinyply::Type::INT64;*/
    /*else if (kind == 'u' && itemsize == 8)  return tinyply::Type::UINT64;*/
    else
    {
        throw std::runtime_error("numpy_dtype_to_tinyply_type: Unsupported NumPy dtype");
    }
}

inline pybind11::dtype tinyply_type_to_numpy_dtype(tinyply::Type t)
{
    using namespace tinyply;
    switch (t)
    {
    case Type::FLOAT32: return pybind11::dtype::of<float>();
    // case Type::FLOAT64: return pybind11::dtype::of<double>();
    case Type::INT8:    return pybind11::dtype("int8");
    case Type::UINT8:   return pybind11::dtype("uint8");
    case Type::INT16:   return pybind11::dtype("int16");
    case Type::UINT16:  return pybind11::dtype("uint16");
    case Type::INT32:   return pybind11::dtype("int32");
    case Type::UINT32:  return pybind11::dtype("uint32");
    /*case Type::INT64:   return pybind11::dtype("int64");*/
    /*case Type::UINT64:  return pybind11::dtype("uint64");*/
    default:
        throw std::runtime_error("tinyply_type_to_numpy_dtype: Unsupported tinyply type");
    }
}

//-----------------------------------------------------------------
// read_ply: Reads a .ply file into a dict[element_name][property_name] -> NumPy array
//-----------------------------------------------------------------
pybind11::dict read_ply(const std::string &filename)
{
    namespace py = pybind11;
    using namespace tinyply;

    // Prepare to read from disk
    std::unique_ptr<std::istream> file_stream;
    file_stream.reset(new std::ifstream(filename, std::ios::binary));
    if (!file_stream || file_stream->fail())
        throw std::runtime_error("Failed to open file: " + filename);

    PlyFile file;
    file.parse_header(*file_stream);

    // We will store the data buffers requested from tinyply here
    // Key: (elementName, propertyName) -> PlyData
    std::map<std::pair<std::string, std::string>, std::shared_ptr<PlyData>> requestedData;

    // For each element, request each property. Because we don't know them upfront,
    // we rely on the element's property list from the parsed header. We must handle
    // potential list or scalar properties.
    for (auto &elm : file.get_elements())
    {
        for (auto &prop : elm.properties)
        {
            // We don't know if it's a list with a certain fixed size or variable size.
            // For fixed list size N, we can pass N to request_properties_from_element.
            // For unknown variable size, pass 0. tinyply will figure out the size from the data.
            // In practice, face/vertex_indices is a common fixed 3 or 4, but can vary.
            // As a generic fallback, we pass 0 to handle variable. If you know it, pass N for speed.
            size_t list_size_hint = 0;
            if (prop.isList && prop.listCount > 0)
            {
                // tinyply will fill in prop.listCount if the header declares a fixed size.
                list_size_hint = prop.listCount;
            }

            try
            {
                auto dataHandle = file.request_properties_from_element(
                    elm.name,
                    { prop.name },
                    list_size_hint
                );
                requestedData[{elm.name, prop.name}] = dataHandle;
            }
            catch(const std::exception &e)
            {
                // The property might fail to load if, for example, a second "colors" property
                // is requested but not actually present. We'll just ignore in that case.
                // Or we can print a warning:
                // std::cerr << "Warning: " << e.what() << std::endl;
            }
        }
    }

    // Actually read the data
    file.read(*file_stream);

    // Now convert to Python dictionaries
    py::dict rootDict; // top-level dict: element -> (property -> np.array)

    for (auto &elm : file.get_elements())
    {
        py::dict propertyDict;
        for (auto &prop : elm.properties)
        {
            auto it = requestedData.find({elm.name, prop.name});
            if (it == requestedData.end()) continue; // wasn't requested or not present
            auto plyData = it->second;
            if (!plyData) continue; // property wasn't found or read

            // Create a NumPy array for this property. We'll interpret:
            //   count = how many "rows" (i.e. element count)
            //   if it's a list property with a known (or constant) list length L, shape will be (count, L)
            //   if it's a scalar property, shape will be (count,).
            const size_t count = plyData->count;
            const tinyply::Type t = plyData->t;
            const size_t numBytes = plyData->buffer.size_bytes();
            const size_t stride   = tinyply::PropertyTable[t].stride;

            // Number of total elements in the buffer = numBytes / stride
            // But we also have to consider if it's a list property. If the list size is uniform, 
            // then each "row" of the property has L items. The PlyData struct does not store that 
            // L explicitly if it's variable, so we do a best guess or use prop.listCount if not zero.
            size_t list_count = 1; // default for scalar
            if (prop.isList)
            {
                if (prop.listCount > 0)
                    list_count = prop.listCount;
                else
                {
                    // If it's truly variable, we can't easily store it in a 2D array. 
                    // You could store them contiguously and keep an index array, or do
                    // an 'object' array. Here we simply skip or throw:
                    throw std::runtime_error("Variable-size list property '" + prop.name 
                        + "' not handled");
                }
            }

            // Now create the appropriate NumPy array
            // shape is either (count,) or (count, list_count)
            py::dtype npDtype = tinyply_type_to_numpy_dtype(t);
            if (list_count == 1)
            {
                // Scalar property
                auto capsule = py::capsule(new uint8_t[numBytes],
                                           [](void *p) { delete[] reinterpret_cast<uint8_t*>(p); });
                std::memcpy(capsule, plyData->buffer.get(), numBytes);

                py::array array(npDtype, {static_cast<py::ssize_t>(count)}, 
                                {static_cast<py::ssize_t>(stride)}, 
                                capsule);
                propertyDict[py::str(prop.name.c_str())] = array;
            }
            else
            {
                // Fixed list property, shape (count, list_count)
                const size_t totalCount = count * list_count;
                const size_t requiredBytes = totalCount * stride;
                if (requiredBytes != numBytes)
                {
                    throw std::runtime_error("Mismatch in fixed list property for " + prop.name);
                }
                auto capsule = py::capsule(new uint8_t[numBytes],
                                           [](void *p) { delete[] reinterpret_cast<uint8_t*>(p); });
                std::memcpy(capsule, plyData->buffer.get(), numBytes);

                py::array array(
                    npDtype,
                    { static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(list_count) },
                    {
                        static_cast<py::ssize_t>(list_count * stride), // stride for each "row"
                        static_cast<py::ssize_t>(stride)               // stride for each "column"
                    },
                    capsule
                );
                propertyDict[py::str(prop.name.c_str())] = array;
            }
        }
        if (propertyDict.size() > 0)
        {
            rootDict[py::str(elm.name.c_str())] = propertyDict;
        }
    }

    return rootDict;
}



//-----------------------------------------------------------------
// write_ply: Writes dict[element_name][property_name] -> NumPy array to a .ply file
//
// We'll treat arrays with shape (N,) as scalar properties, and shape (N,M) as
// a list property of length M. If M=1, we collapse that to scalar. 
// For fully generic usage, you might store an explicit "list_length" array for variable-size lists.
//-----------------------------------------------------------------
void write_ply(const std::string &filename, const pybind11::dict &pyDict, bool isBinary = true)
{
    namespace py = pybind11;
    using namespace tinyply;
    py::module_ np = py::module_::import("numpy");

    PlyFile plyFile;

    // For each element in the dict
    for (auto item : pyDict)
    {
        std::string elementName = py::cast<std::string>(item.first);
        py::dict properties = py::cast<py::dict>(item.second);

        // We'll gather all scalar/list properties under the same element.
        for (auto propItem : properties)
        {
            std::string propertyName = py::cast<std::string>(propItem.first);
            py::array array = py::cast<py::array>(propItem.second);

            // Extract shape info
            auto shape = array.shape();
            auto ndim  = array.ndim();
            size_t count = 0;
            size_t listCount = 1; // default for scalars

            if (ndim == 1)
            {
                // shape (count,)
                count = shape[0];
            }
            else if (ndim == 2)
            {
                // shape (count, listCount)
                count = shape[0];
                listCount = shape[1];
            }
            else
            {
                throw std::runtime_error("write_ply: Only 1D or 2D arrays are supported. ");
            }

            // If listCount == 1, treat it as scalar property
            const auto npDtype = array.dtype();
            tinyply::Type plyType = numpy_dtype_to_tinyply_type(npDtype);

            // We can store the raw pointer
            auto dataPtr = reinterpret_cast<const uint8_t*>(array.data());

            if (listCount == 1)
            {
                // Scalar property
                plyFile.add_properties_to_element(
                    elementName,
                    {propertyName},
                    plyType,             // property data type
                    count,
                    dataPtr,
                    tinyply::Type::INVALID, // list type
                    0                       // list count
                );
            }
            else
            {
                // List property with known length == listCount
                // tinyply requires we specify the list type (the type used for list indices),
                // but in typical .ply usage for faces, we might do something like 
                //   add_properties_to_element("face", {...}, Type::UINT32, faceCount, dataPtr, Type::UINT8, 3 )
                // for 3 indices. The `listCount` must match. For the "list type," we pass e.g. Type::UINT8 
                // for a typical face definition. If you want to vary that, you can adjust accordingly.
                // We'll guess a standard 1-byte list size type (Type::UINT8) for all fixed lists
                plyFile.add_properties_to_element(
                    elementName,
                    { propertyName },
                    plyType,
                    count,
                    dataPtr,
                    tinyply::Type::UINT8,  // or perhaps Type::UINT32 for very large polygons
                    static_cast<uint32_t>(listCount)
                );
            }
        }
    }

    // Add a comment for provenance
    plyFile.get_comments().push_back("Generated by tinyplypy");

    // Write out
    std::filebuf fb;
    std::ios::openmode mode = std::ios::out;
    if (isBinary) mode |= std::ios::binary;
    if (!fb.open(filename, mode))
    {
        throw std::runtime_error("write_ply: failed to open file: " + filename);
    }
    std::ostream outstream(&fb);
    plyFile.write(outstream, isBinary);
}

//-----------------------------------------------------------------
// Pybind11 module definition
//-----------------------------------------------------------------
namespace py = pybind11;

PYBIND11_MODULE(_tinyplypy_binding, m)
{
    m.doc() = "Example tinyply <-> NumPy pybind11 binding";

    m.def("read_ply", &read_ply,
          py::arg("filename"),
          "Reads a PLY file into a Python dict[element][property] -> NumPy array");

    m.def("write_ply", &write_ply,
          py::arg("filename"),
          py::arg("data"),
          py::arg("isBinary") = true,
          "Writes a PLY file from a Python dict[element][property] -> NumPy array. "
          "Pass isBinary=False for ASCII output.");
}


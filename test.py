import numpy as np
import pytest
import os
import tinyplypy

@pytest.mark.parametrize("is_binary", [False, True])
def test_write_read_ply(tmp_path, is_binary):
    """
    Test that writing a dict[element_name][property_name] -> np.array
    then reading it back yields the same data, in both ASCII (is_binary=False)
    and binary (is_binary=True) PLY formats.
    """

    # Create some random data for two elements: e.g. 'vertex' and 'face'
    # The 'vertex' element has scalar properties x, y, z
    # The 'face' element has a list property for the indices with shape (n_faces, 3)
    num_vertices = 5000
    num_faces = 200

    vertex_arrays = {
        "x": np.random.rand(num_vertices).astype(np.float32),
        "y": np.random.rand(num_vertices).astype(np.float32),
        "z": np.random.rand(num_vertices).astype(np.float32)
    }

    keys = "qwertyuiopasdfghjklzxcvbnm"
    tet_arrays = {k: np.random.rand(num_vertices).astype(np.float32) for k in keys}

    # For 2D arrays, shape=(n_faces,3), e.g. triangle indices
    face_arrays = {
        "vertex_indices": np.random.randint(
            low=0, high=num_vertices, size=(num_faces, 3)
        ).astype(np.uint32)
    }

    data_dict = {
        "vertex": vertex_arrays,
        "face": face_arrays,
        "tet": tet_arrays,
    }

    # Construct a filename in the temporary directory
    mode_name = "binary" if is_binary else "ascii"
    ply_file = tmp_path / f"test_{mode_name}.ply"

    # Write the data
    tinyplypy.write_ply(str(ply_file), data_dict, is_binary=is_binary)

    # Read the data back
    loaded_dict = tinyplypy.read_ply(str(ply_file))

    # Now compare shapes and values
    # For each element/property, check that they exist in the loaded data
    for element_name, properties in data_dict.items():
        assert element_name in loaded_dict, (
            f"Element '{element_name}' not found in loaded data"
        )
        loaded_props = loaded_dict[element_name]

        for prop_name, original_array in properties.items():
            assert prop_name in loaded_props, (
                f"Property '{prop_name}' missing under element '{element_name}'"
            )
            loaded_array = loaded_props[prop_name]

            # Check shape
            assert loaded_array.shape == original_array.shape, (
                f"Mismatched shape for {element_name}.{prop_name}.\n"
                f"Original: {original_array.shape}, Loaded: {loaded_array.shape}"
            )

            # Check dtype
            assert loaded_array.dtype == original_array.dtype, (
                f"Mismatched dtype for {element_name}.{prop_name}.\n"
                f"Original: {original_array.dtype}, Loaded: {loaded_array.dtype}"
            )

            # Check values
            # Use np.allclose or np.array_equal depending on whether you want
            # exact integer matches or approximate floating point matches.
            if np.issubdtype(original_array.dtype, np.integer):
                assert np.array_equal(loaded_array, original_array), (
                    f"Mismatched integer values in {element_name}.{prop_name}"
                )
            else:
                # For floating point, allclose is safer.
                np.testing.assert_allclose(
                    loaded_array, original_array,
                    err_msg=f"Mismatched float values in {element_name}.{prop_name}",
                    rtol=1e-5, atol=1e-6
                )

    # Cleanup (optional if you don't want the file left in tmp_path)
    # os.remove(ply_file)  # Usually unnecessary with pytest's tmp_path

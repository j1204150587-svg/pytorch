# Stage the measured changes after op-plugin code generation. In particular,
# StructKernelNpuOpApi.cpp is generated and is absent from the submodule tree.
find_package(Git REQUIRED)

function(patch_embedding_source relative_path)
  set(original "${PROJECT_SOURCE_DIR}/third_party/op-plugin/${relative_path}")
  get_filename_component(filename "${original}" NAME)
  get_filename_component(original_dir "${original}" DIRECTORY)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  set(patch_file "${patch_dir}/${filename}.patch")
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/embedding")
  set(staging_dir "${output_dir}/stage")
  set(output "${output_dir}/${filename}")
  file(MAKE_DIRECTORY "${staging_dir}")

  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${original}" "${staging_dir}/${filename}"
    COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${output_dir}"
            "${GIT_EXECUTABLE}" -c core.autocrlf=false apply --no-index "${patch_file}"
    COMMAND "${CMAKE_COMMAND}" -E rename "${staging_dir}/${filename}" "${output}"
    DEPENDS "${original}" "${patch_file}"
    WORKING_DIRECTORY "${staging_dir}"
    VERBATIM)

  list(FIND OPS_PLUGIN_SRCS "${original}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Missing op-plugin source in the build: ${original}")
  endif()
  list(REMOVE_AT OPS_PLUGIN_SRCS ${position})
  list(INSERT OPS_PLUGIN_SRCS ${position} "${output}")
  set(OPS_PLUGIN_SRCS "${OPS_PLUGIN_SRCS}" PARENT_SCOPE)
  set_property(SOURCE "${output}" APPEND PROPERTY INCLUDE_DIRECTORIES "${original_dir}" "${patch_dir}")
endfunction()

patch_embedding_source(op_plugin/utils/op_api_common.cpp)
patch_embedding_source(op_plugin/ops/opapi/StructKernelNpuOpApi.cpp)

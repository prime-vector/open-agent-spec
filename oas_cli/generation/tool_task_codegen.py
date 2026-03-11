# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Tool-task function generation (DACP tool args + invoke/execute_tool flow)."""

from typing import Any

from .contracts_codegen import _format_contract_for_decorator
from .intelligence_codegen import _generate_intelligence_config
from .task_preamble import _get_task_function_preamble


def _build_tool_args_and_description(
    task_def: dict[str, Any],
) -> tuple[str, list[str], str, dict[str, str]]:
    """Build tool_args lines, tool description, and param mapping for a tool task. Returns (tool_id, tool_args_lines, tool_description_with_params, tool_param_mapping)."""
    tool_id = task_def["tool"]
    tool_params = task_def.get("tool_params", {})
    tool_param_mapping: dict[str, str] = {}
    tool_args_lines: list[str] = []

    if tool_params:
        for param_name, param_info in tool_params.items():
            if isinstance(param_info, dict):
                dacp_param = param_info.get("dacp_param", param_name)
                tool_param_mapping[param_name] = dacp_param
                tool_args_lines.append(f'        "{dacp_param}": {param_name}')
            else:
                tool_param_mapping[param_name] = param_name
                tool_args_lines.append(f'        "{param_name}": {param_name}')
    else:
        input_props = task_def.get("input", {}).get("properties", {})
        for param_name in input_props.keys():
            if tool_id == "file_writer":
                if param_name == "file_path":
                    tool_param_mapping[param_name] = "path"
                    tool_args_lines.append(f'        "path": {param_name}')
                elif param_name == "content":
                    tool_param_mapping[param_name] = "content"
                    tool_args_lines.append(f'        "content": {param_name}')
                else:
                    tool_param_mapping[param_name] = param_name
                    tool_args_lines.append(f'        "{param_name}": {param_name}')
            else:
                tool_param_mapping[param_name] = param_name
                tool_args_lines.append(f'        "{param_name}": {param_name}')

    tool_description = f"Tool: {tool_id}"
    if tool_params:
        param_descriptions = []
        for param_name, param_info in tool_params.items():
            if isinstance(param_info, dict):
                desc = param_info.get("description", param_name)
                param_descriptions.append(f"- {param_name}: {desc}")
            else:
                param_descriptions.append(f"- {param_name}")
        tool_description += "\nParameters:\n" + "\n".join(param_descriptions)
    return tool_id, tool_args_lines, tool_description, tool_param_mapping


def _generate_tool_task_function(
    task_name: str,
    task_def: dict[str, Any],
    spec_data: dict[str, Any],
    agent_name: str,
    memory_config: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Generate a task function that uses a DACP tool."""
    func_name, input_params, output_type, docstring, contract_data = (
        _get_task_function_preamble(
            task_name, task_def, spec_data, agent_name, memory_config
        )
    )
    tool_id, tool_args_lines, tool_description_with_params, tool_param_mapping = (
        _build_tool_args_and_description(task_def)
    )
    decorator = ""
    if contract_data:
        contract_str = _format_contract_for_decorator(contract_data)
        decorator = f"@behavioural_contract(\n    {contract_str}\n)\n"

    return f"""
from dacp import invoke_intelligence, execute_tool
from dacp.protocol import parse_agent_response, is_tool_request, get_tool_request, wrap_tool_result, get_final_response, is_final_response

{decorator}def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Prepare tool arguments
    tool_args = {{
{", ".join(tool_args_lines)}
    }}

    # Create prompt with tool description
    json_example1 = '{{"tool_request": {{"name": "{tool_id}", "args": {{"path": "file_path_here", "content": "content_here"}}}}}}'
    json_example2 = '{{"final_response": {{"result": "your final result here"}}}}'

    tool_prompt = f'''You have access to the following tool:

{tool_description_with_params}

Your task is to use this tool appropriately. You can use the tool by responding with a tool request, or provide a final response.

Available parameters: {list(tool_param_mapping.values())}
Current input values: {{tool_args}}

Respond with JSON in one of these formats:

1. For tool requests:
{{json_example1}}

2. For final responses:
{{json_example2}}

Remember: Only use the tool if it's necessary for your task.'''

    # Configure intelligence for DACP
    intelligence_config = {_generate_intelligence_config(spec_data, config)}

    # Call the LLM with tool context
    response = invoke_intelligence(tool_prompt, intelligence_config)

    # Parse the response
    parsed_response = parse_agent_response(response)

    # Check if LLM wants to use a tool
    if is_tool_request(parsed_response):
        tool_name, tool_params = get_tool_request(parsed_response)

        # Execute the tool
        tool_result = execute_tool(tool_name, tool_params)

        # Wrap the tool result for the LLM
        wrapped_result = wrap_tool_result(tool_name, tool_result)

        # Continue conversation with tool result
        follow_up_prompt = f'''The tool execution result: {{wrapped_result}}

Based on this result, provide your final response in JSON format:

{{{{"final_response": {{{{"result": "your final result here"}}}}}}}}

Remember to respond with valid JSON.'''

        final_response = invoke_intelligence(follow_up_prompt, intelligence_config)
        final_parsed = parse_agent_response(final_response)

        if is_final_response(final_parsed):
            result = get_final_response(final_parsed)
        else:
            result = {{"error": "LLM did not provide final response after tool execution"}}
    else:
        # LLM provided final response directly
        if is_final_response(parsed_response):
            result = get_final_response(parsed_response)
        else:
            result = {{"error": "LLM response format not recognized"}}

    # Map result to expected output format
    if "{tool_id}" == "file_writer":
        # Handle file_writer specific mapping for new DACP response format
        if isinstance(result, dict) and result.get("success") is True:
            # New DACP format: {{'success': True, 'path': '...', 'message': '...'}}
            mapped_result = {{
                "success": True,
                "file_path": result.get("path", tool_args.get("path", "")),
                "bytes_written": len(tool_args.get("content", ""))
            }}
        elif isinstance(result, dict) and "result" in result and ("Written to " in result["result"] or "Successfully wrote" in result["result"]):
            # Legacy format: {{'result': 'Written to path'}} or {{'result': 'Successfully wrote X characters to path'}}
            result_text = result["result"]
            if "Written to " in result_text:
                file_path = result_text.replace("Written to ", "")
            elif "Successfully wrote" in result_text:
                # Extract path from "Successfully wrote X characters to path"
                file_path = result_text.split(" to ")[-1]
            else:
                file_path = tool_args.get("path", "")
            mapped_result = {{
                "success": True,
                "file_path": file_path,
                "bytes_written": len(tool_args.get("content", ""))
            }}
        else:
            mapped_result = {{
                "success": False,
                "file_path": tool_args.get("path", ""),
                "bytes_written": 0
            }}
    else:
        mapped_result = result

    # Return the result in the expected output format
    return {output_type}(**mapped_result)
"""

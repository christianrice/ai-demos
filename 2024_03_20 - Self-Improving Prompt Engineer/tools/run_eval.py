import json
from langchain_core.messages import HumanMessage
from tools.evaluate_prompt_output import (
    evaluate_expected_output_runnable,
    evaluate_bad_output_runnable,
)
from tools.generate_prompt_output import generate_prompt_output_runnable


def evaluate_against_expected_output(expected_output, actual_output):
    evaluation_result = evaluate_expected_output_runnable.invoke(
        {"expected_output": expected_output, "actual_output": actual_output}
    )
    print(
        "Expected Positive Output Result: ",
        evaluation_result.get("eval"),
        evaluation_result.get("failure_reason"),
    )

    return evaluation_result.get("eval"), evaluation_result.get("failure_reason", "NA")


def evaluate_against_bad_output(expected_output, actual_output):
    evaluation_result = evaluate_bad_output_runnable.invoke(
        {"expected_output": expected_output, "actual_output": actual_output}
    )
    print("Expected Negative Output Result: ", evaluation_result.content)

    if evaluation_result.content == "DIFFERENT":
        return True, evaluation_result.content
    else:
        return False, evaluation_result.content


def update_confusion_matrix(
    is_expected_correct,
    is_bad_correct,
    data,
    actual_output,
    expected_evaluation,
    bad_evaluation,
    confusion_matrix,
    inaccurate_responses,
):
    if is_expected_correct:
        confusion_matrix["TP"] += 1
    else:
        confusion_matrix["FN"] += 1
        # Log the inaccurate response details for false negatives
        inaccurate_responses.append(
            {
                "input": data["input"],
                "evaluation_result": expected_evaluation,
            }
        )

    if is_bad_correct:
        confusion_matrix["TN"] += 1
    else:
        confusion_matrix["FP"] += 1
        # Log the inaccurate response details for false positives
        inaccurate_responses.append(
            {
                "input": data["input"],
                "evaluation_result": bad_evaluation,
            }
        )


def extract_arguments(data):
    arguments_list = []

    # Check if data is a dictionary and contains the key 'tool_calls'
    if isinstance(data, dict) and "tool_calls" in data:
        for tool_call in data["tool_calls"]:
            # Extracting the 'arguments' from the tool_call
            function_info = tool_call.get("function", {})
            arguments = function_info.get("arguments", "")

            # If arguments is a string, attempt to parse it as JSON
            if isinstance(arguments, str):
                try:
                    arguments_json = json.loads(arguments)
                    arguments_list.append(arguments_json)
                except json.JSONDecodeError:
                    print("Error decoding JSON from arguments:", arguments)

    return arguments_list


def process_eval_dataset(file_name, prompt_inputs):
    # Initialize the confusion matrix counters
    confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    # Initialize the list to store inaccurate responses
    inaccurate_responses = []

    bad_responses = 0
    with open(file_name, "r") as infile:
        for line_number, line in enumerate(infile, start=1):
            print("\n")
            print(f"Running line {line_number}")
            print("---------------")
            # line = next(infile)
            data = json.loads(line.strip())
            messages = [HumanMessage(content=data.get("input"))]
            memories = data.get("memories", [])

            prompt_inputs["messages"] = messages
            prompt_inputs["memories"] = memories

            response = generate_prompt_output_runnable.invoke(prompt_inputs)
            actual_output = extract_arguments(response.additional_kwargs)

            # Just test the expected output as a control
            # actual_output = data.get("desired_response")

            is_expected_correct, expected_detail = evaluate_against_expected_output(
                data.get("desired_response"), actual_output
            )
            if not is_expected_correct:
                bad_responses += 1

            is_bad_correct, bad_detail = evaluate_against_bad_output(
                data.get("bad_response"), actual_output
            )
            if not is_bad_correct:
                bad_responses += 1

            update_confusion_matrix(
                is_expected_correct,
                is_bad_correct,
                data,
                actual_output,
                expected_detail,
                bad_detail,
                confusion_matrix,
                inaccurate_responses,
            )

            if bad_responses >= 3:
                print("Encountered 3 bad responses. Ending process.")
                break

    print("\n")
    print("Eval Results:")
    print("Confusion Matrix:")
    print(confusion_matrix)

    total_cases = sum(confusion_matrix.values())
    accuracy = (
        (confusion_matrix["TP"] + confusion_matrix["TN"]) / total_cases
        if total_cases
        else 0
    )

    print("\n")
    print("Accuracy: ", accuracy)

    return confusion_matrix, accuracy, inaccurate_responses

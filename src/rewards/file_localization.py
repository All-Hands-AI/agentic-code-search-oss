import ast

def compute_file_f1_score(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

# def file_localization_f1_reward(final_message, instance):
#     predicted_files = set(ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0]))
#     # print("Predicted files:", predicted_files)
#     true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
#     # print("True files:", true_files)
#     return compute_file_f1_score(predicted_files, true_files)

def file_localization_f1_reward(final_message, instance, working_dir=None):

    # <file=full_path1/file1.py>
    # <line>10</line>
    # <class>MyClass1</class>
    # <function>my_function1</function>
    # </file>
    # <file=full_path2/file2.py>
    # <line>76</line>
    # <function>MyClass2.my_function2</function>
    # </file>
    # <file=full_path3/file3.py>
    # <line>24</line>
    # <line>156</line>
    # <function>my_function3</function>
    # </file>

    # Extract only file paths from the final message
    predicted_files = set()
    file_sections = final_message.split("<file=")[1:]
    for section in file_sections:
        file_path = section.split(">")[0]
        if working_dir:
            # Remove the working directory prefix if present
            if file_path.startswith(working_dir):
                file_path = file_path[len(working_dir):]

            if file_path.startswith("/"):
                file_path = file_path[1:]

        predicted_files.add(file_path)
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))

    return list(predicted_files), compute_file_f1_score(predicted_files, true_files)


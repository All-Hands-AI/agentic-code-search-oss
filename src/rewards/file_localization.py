def compute_file_f1_score(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def file_localization_f1_reward(final_message, instance):
    predicted_files = set(ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0]))
    print("Predicted files:", predicted_files)
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    print("True files:", true_files)
    return compute_file_f1_score(predicted_files, true_files)
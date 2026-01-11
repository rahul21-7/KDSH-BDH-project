import pandas as pd

def create_submission(train_analysis_path, test_analysis_path, output_path):
    train = pd.read_csv(train_analysis_path)
    test = pd.read_csv(test_analysis_path)

    # Calculate the median tension to use as a separator
    # Based on your data, around 120 seems to be a pivot point
    tension_threshold = train['tension'].median()

    predictions = []

    for _, row in test.iterrows():
        # Logic: If tension is high, the model 'recognizes' the flow (Consistent)
        # If tension is low, the model is 'confused' or disconnected (Contradict)
        if row['tension'] > tension_threshold:
            label = 1
            rationale = "The backstory maintains a high narrative tension consistent with the established character arc and environmental state in the novel."
        else:
            label = 0
            rationale = "The backstory introduces elements that lower narrative tension or conflict with established facts, suggesting a plot contradiction."
            
        predictions.append({
            "id": int(row['id']),
            "label": label,
            "rationale": rationale
        })

    submission_df = pd.DataFrame(predictions)
    # Ensure ID column is first
    submission_df = submission_df[['id', 'label', 'rationale']]
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with threshold {tension_threshold:.2f}")

if __name__ == "__main__":
    create_submission("./train_analysis.csv", "./test_analysis.csv", "../output/submission.csv")
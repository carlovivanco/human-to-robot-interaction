# !pip install transformers
# !pip install tf-keras

from transformers import pipeline


tools = ["hammer", "screwdriver", "wrench", "pliers"]
actions = ["pass", "use", "take"]


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def extract_tool_and_action(input_text):
    tool_result = classifier(input_text, candidate_labels=tools)
    detected_tool = tool_result['labels'][0]

    action_result = classifier(input_text, candidate_labels=actions)
    detected_action = action_result['labels'][0]

    return detected_tool, detected_action


input_text = "you should be capable of building a house with huge expensive great amazing pliers. But first pass it to me."
tool, action = extract_tool_and_action(input_text)


print(f"Detected Tool: {tool}")
print(f"Detected Action: {action}")

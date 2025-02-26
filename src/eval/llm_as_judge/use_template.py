def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def build_conversation_history(conversation, current_turn):
    history = []
    for i in range(current_turn + 1): 
        history.append(f"用户：{conversation[i]['user']}")
        if i < current_turn: 
            history.append(f"助手：{conversation[i]['assistant']}")
    return "\n".join(history)

def use_judge_template(conversation, reference_answer, generated_answer, current_turn):
    template = load_template('src/config/template/prompt.txt')
    
    convsation_history = build_conversation_history(conversation, current_turn)
    
    return template.replace(
        "{会话过程}", convsation_history
    ).replace(
        "{参考答案}", reference_answer
    ).replace(
        "{AI助手撰写的答案}", generated_answer
    )
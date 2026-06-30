from langchain_aws import ChatBedrockConverse


def demo_chatbot(messages):
    demo_llm = ChatBedrockConverse(
        credentials_profile_name="default",
        model="deepseek.v3-v1:0",
        temperature=0.1,
        max_tokens=1000,
    )

    return demo_llm.invoke(messages)


messages = [
    {
        "role": "user",
        "content": [{"text": "What is the capital of France?"}]
    }
]

response = demo_chatbot(messages)

print(response.content)
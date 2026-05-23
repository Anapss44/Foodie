import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_MODEL_NAME = os.getenv("LOCAL_CHAT_MODEL", "distilgpt2")
USE_OPENAI = bool(OPENAI_API_KEY)
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "CHATBOT_SYSTEM_PROMPT",
    "You are a friendly, chatty AI assistant inside a food delivery app called Foodie. Your job is to help users log food, answer calorie and nutrition questions, and have natural conversations about food, meals, recipes, and healthy choices. Be warm, concise, and encourage follow-up by asking a polite question when appropriate. If meal details are unclear, ask the user to clarify. If a user asks about food, cooking, or general wellness, answer helpfully with examples and practical guidance. If the user asks something outside nutrition, respond politely and steer the conversation back to food, health, or app assistance. Keep responses conversational, helpful, and positive."
)

local_model = None
local_tokenizer = None
local_device = None


def load_local_model():
    global local_model, local_tokenizer, local_device
    if local_model is not None and local_tokenizer is not None:
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        app.logger.warning("Local model dependencies are missing: %s", exc)
        return

    local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    local_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
    local_device = "cuda" if torch.cuda.is_available() else "cpu"
    local_model.to(local_device)
    local_model.eval()


def build_local_prompt(system, messages):
    prompt_parts = []
    if system:
        prompt_parts.append(system.strip())
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        if role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(f"User: {content}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def generate_local_reply(system, messages):
    load_local_model()
    if local_model is None or local_tokenizer is None:
        raise RuntimeError("Local model is unavailable. Install transformers and torch or set OPENAI_API_KEY.")

    import torch

    effective_system = system.strip() if isinstance(system, str) and system.strip() else DEFAULT_SYSTEM_PROMPT
    prompt = build_local_prompt(effective_system, messages)
    inputs = local_tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(local_device)
    attention_mask = inputs.attention_mask.to(local_device)

    output_ids = local_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.85,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=local_tokenizer.eos_token_id
    )
    generated_text = local_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = generated_text[len(prompt):].strip()
    reply = reply.split("User:", 1)[0].strip()
    if not reply:
        reply = "I'm ready to help you with calories and nutrition. Ask me anything!"
    return reply


def generate_openai_reply(system, messages):
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("OpenAI package is not installed: %s" % exc)

    openai.api_key = OPENAI_API_KEY
    effective_system = system.strip() if isinstance(system, str) and system.strip() else DEFAULT_SYSTEM_PROMPT
    chat_messages = [{"role": "system", "content": effective_system}]
    for message in messages:
        role = message.get("role", "user")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        chat_messages.append({"role": role, "content": message.get("content", "")})

    response = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=chat_messages,
        temperature=0.7,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    system = payload.get("system", "")
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be an array"}), 400

    try:
        if USE_OPENAI:
            reply = generate_openai_reply(system, messages)
        else:
            reply = generate_local_reply(system, messages)
        return jsonify({"reply": reply})
    except Exception as exc:
        app.logger.exception("Chat generation failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Python chatbot backend is running."})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

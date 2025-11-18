#%%
import requests
import json
import base64
from pathlib import Path
from transformers import AutoTokenizer
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:12b"

# Load the tokenizer for decoding (only needed for Example 8)
# Note: This will download the tokenizer on first run
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

#%%
# =============================================================================
# EXAMPLE 1: Ollama Native API - Simple Generate
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 1: Ollama Native API - Simple Generate")
print("*" * 40)

url = f"{OLLAMA_BASE_URL}/api/generate"
payload = {
    "model": MODEL_NAME,
    "prompt": "hello",
    "stream": False  # Set to False to get complete response at once
}

print("\n📤 REQUEST:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE:")
print(json.dumps(result, indent=2))

#%%
# =============================================================================
# EXAMPLE 2: Ollama Native API - Streaming Response
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 2: Ollama Native API - Streaming Response")
print("*" * 40)

url = f"{OLLAMA_BASE_URL}/api/generate"
payload = {
    "model": MODEL_NAME,
    "prompt": "Tell me a short joke about programming.",
    "stream": True
}

print("\n📤 REQUEST:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

print("\n📥 RESPONSE (streaming chunks):")
response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(json.dumps(chunk, indent=2))
        if chunk.get("done", False):
            break

#%%
# =============================================================================
# EXAMPLE 3: Ollama Native API - Conversation with Context
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 3: Ollama Native API - Conversation with Context")
print("*" * 40)

# First message in conversation
url = f"{OLLAMA_BASE_URL}/api/generate"
payload = {
    "model": MODEL_NAME,
    "prompt": "My name is Alice and I'm learning Python.",
    "stream": False
}

print("\n📤 REQUEST #1:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE #1:")
print(json.dumps(result, indent=2))

# Save the context for the next message
context = result.get("context")

print("\n" + "─" * 80)

# Second message - continuing the conversation
payload = {
    "model": MODEL_NAME,
    "prompt": "What was my name again?",
    "context": context,  # Include previous context
    "stream": False
}

print("\n📤 REQUEST #2 (with context from previous call):")
print(f"POST {url}")
# Don't print the full context array as it's very long, just indicate it's there
payload_display = payload.copy()
payload_display["context"] = f"<array of {len(context)} tokens>" if context else None
print(json.dumps(payload_display, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE #2:")
print(json.dumps(result, indent=2))

# Third message - continuing further
context = result.get("context")

print("\n" + "─" * 80)

payload = {
    "model": MODEL_NAME,
    "prompt": "What programming language did I mention?",
    "context": context,
    "stream": False
}

print("\n📤 REQUEST #3 (with context from previous call):")
print(f"POST {url}")
payload_display = payload.copy()
payload_display["context"] = f"<array of {len(context)} tokens>" if context else None
print(json.dumps(payload_display, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE #3:")
print(json.dumps(result, indent=2))

#%%
# =============================================================================
# EXAMPLE 4: OpenAI-Compatible API - Simple Chat
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 4: OpenAI-Compatible API - Simple Chat")
print("*" * 40)

url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": False
}

print("\n📤 REQUEST:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE:")
print(json.dumps(result, indent=2))

#%%
# =============================================================================
# EXAMPLE 5: OpenAI-Compatible API - Multi-turn Conversation
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 5: OpenAI-Compatible API - Multi-turn Conversation")
print("*" * 40)

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant that specializes in explaining technical concepts."}
]

# First turn
user_message = "What is a REST API?"
conversation_history.append({"role": "user", "content": user_message})

url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
payload = {
    "model": MODEL_NAME,
    "messages": conversation_history,
    "stream": False
}

print("\n📤 REQUEST #1:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE #1:")
print(json.dumps(result, indent=2))

assistant_message = result['choices'][0]['message']['content']

# Add assistant's response to history
conversation_history.append({"role": "assistant", "content": assistant_message})

print("\n" + "─" * 80)

# Second turn
user_message = "Can you give me a simple example?"
conversation_history.append({"role": "user", "content": user_message})

payload = {
    "model": MODEL_NAME,
    "messages": conversation_history,
    "stream": False
}

print("\n📤 REQUEST #2 (with full conversation history):")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE #2:")
print(json.dumps(result, indent=2))

assistant_message = result['choices'][0]['message']['content']

# Add assistant's response to history
conversation_history.append({"role": "assistant", "content": assistant_message})

print("\n" + "─" * 80)

# Third turn
user_message = "What HTTP methods are commonly used?"
conversation_history.append({"role": "user", "content": user_message})

payload = {
    "model": MODEL_NAME,
    "messages": conversation_history,
    "stream": False
}

print("\n📤 REQUEST #3 (with full conversation history):")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE #3:")
print(json.dumps(result, indent=2))

#%%
# =============================================================================
# EXAMPLE 6: OpenAI-Compatible API - Streaming Response
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 6: OpenAI-Compatible API - Streaming Response")
print("*" * 40)

url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Write a haiku about coding."}
    ],
    "stream": True
}

print("\n📤 REQUEST:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

print("\n📥 RESPONSE (SSE streaming chunks):")
response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        print(f"Raw: {line_text}")
        if line_text.startswith("data: "):
            data = line_text[6:]  # Remove "data: " prefix
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                print(f"Parsed: {json.dumps(chunk, indent=2)}")
            except json.JSONDecodeError:
                continue

#%%
# =============================================================================
# EXAMPLE 7: Native API - Advanced Options
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 7: Ollama Native API - Advanced Options")
print("*" * 40)

url = f"{OLLAMA_BASE_URL}/api/generate"
payload = {
    "model": MODEL_NAME,
    "prompt": "Explain quantum computing in one sentence.",
    "stream": False,
    "options": {
        "temperature": 0.7,  # Controls randomness (0-1)
        "top_p": 0.9,        # Nucleus sampling
        "top_k": 40,         # Top-k sampling
        "num_predict": 100,  # Maximum tokens to generate
    }
}

print("\n📤 REQUEST:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE:")
print(json.dumps(result, indent=2))

#%%
# =============================================================================
# EXAMPLE 8: Decoding the Context Array
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 8: Decoding the Context Array - Understanding Token IDs")
print("*" * 40)

# First, let's make a simple call to get a context array
url = f"{OLLAMA_BASE_URL}/api/generate"
payload = {
    "model": MODEL_NAME,
    "prompt": "Hi!",
    "stream": False
}

print("\n📤 REQUEST:")
print(f"POST {url}")
print(json.dumps(payload, indent=2))

response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE:")
print(json.dumps(result, indent=2))

# Decode the context array using the tokenizer
context = result['context']

print("\n" + "─" * 80)
print("\n🔍 DECODING TOKEN IDS WITH GEMMA3 TOKENIZER:")
print(f"\nContext contains {len(context)} tokens")
print(f"\nRaw token IDs: {context}")

# Decode the full context
decoded_text = tokenizer.decode(context)
print(f"\n📝 Full decoded text:\n{repr(decoded_text)}")

# Decode each token individually to show special tokens
print("\n" + "─" * 80)
print("\n🔬 TOKEN-BY-TOKEN BREAKDOWN:")
print(f"{'Index':<8} {'Token ID':<12} {'Token Text':<40} {'Special?':<15}")
print("─" * 80)

for i, token_id in enumerate(context):
    # Decode single token
    token_text = tokenizer.decode([token_id])
    
    # Check if it's a special token - check both by ID and by text pattern
    is_special = (token_id in tokenizer.all_special_ids or 
                  token_text in tokenizer.all_special_tokens or
                  (token_text.startswith('<') and token_text.endswith('>')))
    special_name = ""
    
    if is_special:
        # Identify specific special tokens
        if '<start_of_turn>' in token_text:
            special_name = "START_OF_TURN"
        elif '<end_of_turn>' in token_text:
            special_name = "END_OF_TURN"
        elif token_id == tokenizer.bos_token_id:
            special_name = "BOS (start)"
        elif token_id == tokenizer.eos_token_id:
            special_name = "EOS (end)"
        elif token_id == tokenizer.pad_token_id:
            special_name = "PAD"
        else:
            special_name = "SPECIAL"
    
    # Format token text for display (show whitespace explicitly)
    display_text = repr(token_text) if is_special or not token_text.strip() else token_text
    
    print(f"{i:<8} {token_id:<12} {display_text:<40} {special_name:<15}")

#%%
# =============================================================================
# EXAMPLE 9: OpenAI-Compatible API - Complete Example with Image, All Roles & Sampling
# =============================================================================
print("\n" + "*" * 40)
print("EXAMPLE 9: OpenAI-Compatible API - Vision with All Features")
print("*" * 40)

# Load and display the image
image_path = Path("assets/img1.jpg")
img = Image.open(image_path)

print(f"\n📸 Loading image: {image_path}")
print(f"Image size: {img.size} (width x height)")
print(f"Image mode: {img.mode}")

# Display image inline (for Jupyter/IPython environments)
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')
plt.title(f"Input Image: {image_path.name}")
plt.tight_layout()
plt.show()

# Encode image to base64
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

print(f"\n✅ Image encoded to base64 ({len(img_base64)} characters)")

# Prepare the request with all bells and whistles
url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "system",
            "content": "You are an expert image analyst with deep knowledge in computer vision, object detection, and scene understanding. Provide detailed, accurate, and structured analysis of images."
        },
        {
            "role": "developer",
            "content": "When labeling images, always provide: 1) Main objects/subjects, 2) Scene context, 3) Notable details, 4) Colors and composition. Be concise but thorough."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze and label this image. Describe what you see, including objects, scene, colors, and any notable details."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                }
            ]
        }
    ],
    "stream": False,
    # Sampling parameters - all the bells and whistles
    "temperature": 0.7,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
    "top_p": 0.9,           # Nucleus sampling (consider top 90% probability mass)
    "top_k": 40,            # Consider top 40 tokens
    "max_tokens": 500,      # Maximum tokens to generate
    "frequency_penalty": 0.0,  # Reduce repetition (-2.0 to 2.0)
    "presence_penalty": 0.0,   # Encourage new topics (-2.0 to 2.0)
    "stop": None,           # Stop sequences (e.g., ["\n\n", "END"])
}

print("\n📤 REQUEST:")
print(f"POST {url}")
# Don't print the full base64 image, just show structure
payload_display = payload.copy()
payload_display["messages"] = [
    payload["messages"][0],  # system
    payload["messages"][1],  # developer
    {
        "role": "user",
        "content": [
            payload["messages"][2]["content"][0],  # text
            {
                "type": "image_url",
                "image_url": {"url": f"<base64 encoded image: {len(img_base64)} chars>"}
            }
        ]
    }
]
print(json.dumps(payload_display, indent=2))

print("\n⏳ Sending request with image...")
response = requests.post(url, json=payload)
result = response.json()

print("\n📥 RESPONSE:")
# Pretty print the response
if "choices" in result:
    print(f"\nModel: {result.get('model', 'N/A')}")
    print(f"Created: {result.get('created', 'N/A')}")
    print(f"\nAnalysis:")
    print("─" * 80)
    print(result['choices'][0]['message']['content'])
    print("─" * 80)
    
    if "usage" in result:
        print(f"\n📊 Token Usage:")
        print(f"  Prompt tokens: {result['usage'].get('prompt_tokens', 'N/A')}")
        print(f"  Completion tokens: {result['usage'].get('completion_tokens', 'N/A')}")
        print(f"  Total tokens: {result['usage'].get('total_tokens', 'N/A')}")
    
    print(f"\nFinish reason: {result['choices'][0].get('finish_reason', 'N/A')}")
else:
    # Print full response if structure is different (e.g., error)
    print(json.dumps(result, indent=2))


#%%


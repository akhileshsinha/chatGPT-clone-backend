from transformers import pipeline


class QwenModel:
    def __init__(self):
        self.pipe = None

    def load(self):

        print("Loading Qwen3-4B...")

        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen3-4B",
            device="mps",
        )

        print("Qwen3-4B loaded.")

    
    def generate(self, prompt):
        if self.pipe is None:
            self.load()

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        prompt_text = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        result = self.pipe(
            prompt_text,
            max_new_tokens=512,
            do_sample=False,
        )

        response = result[0]["generated_text"]

        return response[len(prompt_text):].strip()

    def unload(self):

        if self.pipe is None:
            return

        print("Unloading Qwen3-4B...")

        del self.pipe
        self.pipe = None
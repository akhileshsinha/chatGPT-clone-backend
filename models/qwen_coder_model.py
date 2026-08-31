from transformers import pipeline


class QwenCoderModel:

    MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

    def __init__(self):
        self.pipe = None

    def load(self):
        print("Loading Qwen2.5-Coder-7B-Instruct...")

        self.pipe = pipeline(
            "text-generation",
            model=self.MODEL_NAME,
            device="mps",
        )

        print("Qwen2.5-Coder-7B-Instruct loaded.")

    def generate(self, prompt):

        if self.pipe is None:
            self.load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineer. "
                    "Provide accurate, production-quality code. "
                    "Explain the solution clearly when appropriate. "
                    "Use Markdown code blocks for code."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        result = self.pipe(
            prompt_text,
            max_new_tokens=2048,
        )

        response = result[0]["generated_text"]

        return response[len(prompt_text):].strip()

    def unload(self):

        if self.pipe is None:
            return

        print("Unloading Qwen2.5-Coder-7B-Instruct...")

        del self.pipe
        self.pipe = None
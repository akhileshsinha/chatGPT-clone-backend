from transformers import pipeline


class QwenCoderModel:

    def parse_response(self, response):
        explanation = ""
        code = ""

        if "EXPLANATION:" in response:
            response = response.split(
            "EXPLANATION:",
            1
            )[1]

        if "CODE:" in response:
            explanation_part, code_part = response.split(
                "CODE:",
                1
            )

            explanation = explanation_part.strip()

            code_part = code_part.strip()

            if "```" in code_part:
                parts = code_part.split("```")

                if len(parts) >= 2:
                    code = parts[1]

                    # Remove language identifier
                    lines = code.splitlines()

                    if lines and lines[0].strip().lower() in {
                        "javascript",
                        "typescript",
                        "js",
                        "ts",
                        "jsx",
                        "tsx",
                        "python",
                        "java",
                        "kotlin",
                        "json",
                        "css",
                        "html",
                    }:
                        lines = lines[1:]

                        code = "\n".join(lines).strip()
                    else:
                        code = code_part

            else:
                code = code_part

        else:
            explanation = response

        return {
            "explanation": explanation,
            "code": code,
        }


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
                    "Analyze the user's request and provide a production-quality solution.\n\n"
                    "Your response MUST follow this format:\n\n"
                    "EXPLANATION:\n"
                    "<brief explanation>\n\n"
                    "CODE:\n"
                    "```language\n"
                    "<complete code>\n"
                    "```\n\n"
                    "Do not put anything after the CODE section."
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
        generated = result[0]["generated_text"]
        response = generated[len(prompt_text):].strip()
        return self.parse_response(response)

    def unload(self):

        if self.pipe is None:
            return

        print("Unloading Qwen2.5-Coder-7B-Instruct...")

        del self.pipe
        self.pipe = None
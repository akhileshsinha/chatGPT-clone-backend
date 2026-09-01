import gc
import json
import re

from transformers import pipeline


class QwenCoderModel:

    MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

    def __init__(self):
        self.pipe = None

    # ============================================================
    # Load model
    # ============================================================

    def load(self):

        if self.pipe is not None:
            return

        print(
            "Loading Qwen2.5-Coder-7B-Instruct..."
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.MODEL_NAME,
            device="mps",
        )

        print(
            "Qwen2.5-Coder-7B-Instruct loaded."
        )

    # ============================================================
    # Existing code-generation response parser
    # ============================================================

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

                    lines = code.splitlines()

                    if (
                            lines
                            and lines[0].strip().lower()
                            in {
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
                    }
                    ):
                        lines = lines[1:]

                    code = "\n".join(
                        lines
                    ).strip()

            else:

                code = code_part

        else:

            explanation = response

        return {
            "explanation": explanation,
            "code": code,
        }

    # ============================================================
    # Existing single-code generation
    # ============================================================

    def generate(self, prompt):

        if self.pipe is None:
            self.load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineer. "
                    "Analyze the user's request and provide "
                    "a production-quality solution.\n\n"

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

        prompt_text = (
            self.pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        result = self.pipe(
            prompt_text,
            max_new_tokens=2048,
            do_sample=False,
        )

        generated = result[0][
            "generated_text"
        ]

        response = generated[
            len(prompt_text):
        ].strip()

        return self.parse_response(
            response
        )

    # ============================================================
    # Project generation
    # ============================================================

    def generate_project(self, prompt):

        if self.pipe is None:
            self.load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software architect "
                    "and autonomous coding agent.\n\n"

                    "Your task is to design a complete software "
                    "project based on the user's request.\n\n"

                    "The user may ask you to create applications "
                    "using technologies such as React, Node.js, "
                    "Python, Express, TypeScript, etc.\n\n"

                    "Return ONLY valid JSON.\n"
                    "Do NOT use Markdown.\n"
                    "Do NOT wrap the JSON in ```json fences.\n\n"

                    "The JSON MUST have exactly this general structure:\n\n"

                    "{\n"
                    '  "message": "short explanation",\n'
                    '  "actions": [\n'
                    "    {\n"
                    '      "type": "create_file",\n'
                    '      "path": "relative/path/to/file",\n'
                    '      "content": "complete file content"\n'
                    "    }\n"
                    "  ]\n"
                    "}\n\n"

                    "Rules:\n"

                    "1. Use ONLY relative file paths.\n"

                    "2. Never use absolute paths.\n"

                    "3. Never use paths beginning with /.\n"

                    "4. Never use paths containing ../.\n"

                    "5. Every create_file action must contain "
                    "the COMPLETE file content.\n"

                    "6. Include all important files required "
                    "for the requested project.\n"

                    "7. Include package.json when the project "
                    "requires npm dependencies.\n"

                    "8. Include configuration files when they "
                    "are required to run the project.\n"

                    "9. Do not create unnecessary files.\n"

                    "10. Do not execute commands yourself.\n"

                    "11. Do not describe commands instead of "
                    "creating the required files.\n"

                    "12. Do not return Markdown.\n"

                    "13. Do not return code fences.\n"

                    "14. Do not put comments or explanations "
                    "outside the JSON.\n"

                    "15. The output MUST be valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = (
            self.pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        result = self.pipe(
            prompt_text,
            max_new_tokens=4096,
            do_sample=False,
        )

        generated = result[0][
            "generated_text"
        ]

        response = generated[
            len(prompt_text):
        ].strip()

        return self.parse_project_response(
            response
        )

    # ============================================================
    # Project response parser
    # ============================================================

    def parse_project_response(
            self,
            response
    ):

        cleaned = response.strip()

        # --------------------------------------------------------
        # Remove Markdown JSON fences if model accidentally adds
        # them.
        # --------------------------------------------------------

        if cleaned.startswith("```"):

            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            cleaned = re.sub(
                r"\s*```$",
                "",
                cleaned,
            )

        cleaned = cleaned.strip()

        # --------------------------------------------------------
        # Direct JSON parsing
        # --------------------------------------------------------

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            # ----------------------------------------------------
            # Try extracting JSON object from surrounding text
            # ----------------------------------------------------

            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end == -1:

                raise ValueError(
                    "Qwen did not return valid project JSON."
                )

            json_text = cleaned[
                start:end + 1
            ]

            try:

                data = json.loads(
                    json_text
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    "Unable to parse Qwen project response "
                    f"as JSON: {error}"
                )

        # --------------------------------------------------------
        # Validate root object
        # --------------------------------------------------------

        if not isinstance(data, dict):

            raise ValueError(
                "Project response must be a JSON object."
            )

        # --------------------------------------------------------
        # Message
        # --------------------------------------------------------

        message = data.get(
            "message",
            "",
        )

        if not isinstance(
                message,
                str,
        ):
            message = str(
                message
            )

        # --------------------------------------------------------
        # Actions
        # --------------------------------------------------------

        actions = data.get(
            "actions",
            [],
        )

        if not isinstance(
                actions,
                list,
        ):

            raise ValueError(
                "Project actions must be an array."
            )

        validated_actions = []

        for action in actions:

            if not isinstance(
                    action,
                    dict,
            ):
                continue

            action_type = action.get(
                "type"
            )

            path = action.get(
                "path"
            )

            content = action.get(
                "content",
                "",
            )

            # ----------------------------------------------------
            # Currently we only support create_file.
            # ----------------------------------------------------

            if action_type != "create_file":
                continue

            if not isinstance(
                    path,
                    str,
            ):
                continue

            path = path.strip()

            if not path:
                continue

            # ----------------------------------------------------
            # Security validation
            # ----------------------------------------------------

            if path.startswith("/"):
                continue

            if path.startswith("\\"):
                continue

            normalized_path = path.replace(
                "\\",
                "/",
            )

            if ".." in normalized_path.split("/"):

                continue

            # ----------------------------------------------------
            # Normalize content
            # ----------------------------------------------------

            if not isinstance(
                    content,
                    str,
            ):

                content = str(
                    content
                )

            validated_actions.append(
                {
                    "type": "create_file",
                    "path": normalized_path,
                    "content": content,
                }
            )

        # --------------------------------------------------------
        # Return structured project
        # --------------------------------------------------------

        return {
            "message": message,
            "actions": validated_actions,
        }

    # ============================================================
    # Unload model
    # ============================================================

    def unload(self):

        if self.pipe is None:
            return

        print(
            "Unloading Qwen2.5-Coder-7B-Instruct..."
        )

        del self.pipe

        self.pipe = None

        gc.collect()

        print(
            "Qwen2.5-Coder-7B-Instruct unloaded."
        )
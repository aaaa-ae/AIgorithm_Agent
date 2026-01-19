#!/usr/bin/env python3
"""
算法导论题库生成Pipeline
从markdown文件提取题目并生成答案的完整流程
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import re
import yaml


@dataclass
class PipelineConfig:
    """Pipeline配置类"""
    # API配置
    api_key: Optional[str] = None
    api_base: str = "https://api.siliconflow.cn/v1/chat/completions"
    model_name: str = "Qwen/Qwen3-32B"

    # 生成参数
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 700

    # 重试和限速
    retry_times: int = 3
    request_timeout: int = 180
    request_interval: float = 2.0

    # 文件路径
    input_md_path: str = ""
    output_dir: str = ""

    # 处理选项
    generate_answers: bool = True
    save_checkpoint: bool = True
    checkpoint_interval: int = 10  # 每处理N个题目保存一次checkpoint

    # 配置文件路径
    config_file: str = "/home/guoziyang/AIgorithm_Agent/config/base2.yaml"

    @classmethod
    def from_config_file(cls, config_file: str = None, **kwargs) -> 'PipelineConfig':
        """从配置文件加载配置"""
        if config_file is None:
            config_file = "/home/guoziyang/AIgorithm_Agent/config/base2.yaml"

        # 默认配置
        default_config = {
            'api_key': None,
            'api_base': "https://api.siliconflow.cn/v1/chat/completions",
            'model_name': "Qwen/Qwen3-32B",
            'temperature': 0.3,
            'top_p': 0.9,
            'max_tokens': 700,
            'retry_times': 3,
            'request_timeout': 180,
            'request_interval': 2.0,
            'generate_answers': True,
            'save_checkpoint': True,
            'checkpoint_interval': 10,
        }

        # 如果配置文件存在，读取它
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                # 获取LLM配置
                llm_config = config_data.get('llm', {})
                use_provider = llm_config.get('use', 'siliconflow')
                provider_config = llm_config.get(use_provider, {})

                # 从配置文件更新默认配置
                if provider_config:
                    default_config.update({
                        'api_key': provider_config.get('api_key'),
                        'api_base': provider_config.get('api_base', ''),
                        'model_name': provider_config.get('model', ''),
                        'temperature': provider_config.get('temperature', 0.3),
                        'top_p': provider_config.get('top_p', 0.9),
                        'max_tokens': provider_config.get('max_tokens', 700),
                    })

                    # 处理API base URL格式
                    if default_config['api_base'] and not default_config['api_base'].endswith('/chat/completions'):
                        if 'dmxapi' in default_config['api_base']:
                            default_config['api_base'] = default_config['api_base'] + '/chat/completions'
                        elif 'siliconflow' in default_config['api_base']:
                            default_config['api_base'] = default_config['api_base'] + '/chat/completions'
                        elif 'deepseek' in default_config['api_base']:
                            default_config['api_base'] = default_config['api_base'] + '/chat/completions'

                # 保存配置文件路径
                default_config['config_file'] = config_file

            except Exception as e:
                print(f"警告：读取配置文件失败: {e}")
                print("使用默认配置")

        # 用kwargs覆盖配置（命令行参数优先）
        default_config.update(kwargs)

        # 创建配置实例
        return cls(**default_config)


class ExerciseExtractor:
    """题目提取器（从extract_exercises.py移植并优化）"""

    def __init__(self):
        self.exercises = []
        self.problems = []

    def load_file(self, file_path: str) -> str:
        """加载markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        except Exception as e:
            raise Exception(f"读取文件时出错: {e}")

    def extract_exercises(self, content: str) -> List[Dict[str, Any]]:
        """提取练习题"""
        exercises = []

        # 匹配练习题模式
        exercise_pattern = r'(\d+\.\d+-\d+)\s+([^.\n]*(?:\.[^.\n]*)*)?\.?\s*([^.\n]*(?:\.[^.\n]*)*)?\.?'
        exercise_pattern_star = r'\*(\d+\.\d+-\d+)\s+([^.\n]*(?:\.[^.\n]*)*)?\.?\s*([^.\n]*(?:\.[^.\n]*)*)?\.?'

        lines = content.split('\n')
        current_chapter = ""

        for line in lines:
            line = line.strip()

            # 检测章节标题
            if line.startswith('# 第') and '章' in line:
                current_chapter = line.replace('#', '').strip()
                continue

            # 匹配普通练习题
            match = re.match(exercise_pattern, line)
            if match:
                exercise_id = match.group(1)
                question_start = line.find(exercise_id) + len(exercise_id)
                question = line[question_start:].strip()

                if question and not question.startswith('.'):
                    exercises.append({
                        'id': exercise_id,
                        'chapter': current_chapter,
                        'question': question,
                        'type': 'exercise'
                    })
                continue

            # 匹配带星号的练习题
            match_star = re.match(exercise_pattern_star, line)
            if match_star:
                exercise_id = match_star.group(1)
                question_start = line.find(exercise_id) + len(exercise_id)
                question = line[question_start:].strip()

                if question and not question.startswith('.'):
                    exercises.append({
                        'id': exercise_id,
                        'chapter': current_chapter,
                        'question': question,
                        'type': 'exercise'
                    })

        return exercises

    def extract_problems(self, content: str) -> List[Dict[str, Any]]:
        """提取思考题"""
        problems = []
        lines = content.split('\n')
        current_chapter = ""
        current_problem = None

        for line in lines:
            line = line.strip()

            # 检测章节标题
            if line.startswith('# 第') and '章' in line:
                current_chapter = line.replace('#', '').strip()
                continue

            # 匹配思考题标题
            if re.match(r'#\s*\d+-\d+', line):
                # 保存上一个思考题
                if current_problem:
                    problems.append(current_problem)

                # 开始新的思考题
                problem_id = re.search(r'(\d+-\d+)', line).group(1)
                title = line.replace('#', '').strip()

                current_problem = {
                    'id': problem_id,
                    'type': 'problem',
                    'chapter': current_chapter,
                    'title': title,
                    'content': [],
                    'full_text': []
                }
                continue

            # 如果当前有思考题，收集内容
            if current_problem and line and not line.startswith('#'):
                if line.startswith('[') or line.startswith('Omitted'):
                    continue

                current_problem['content'].append(line)
                current_problem['full_text'].append(line)

        # 保存最后一个思考题
        if current_problem:
            problems.append(current_problem)

        # 清理思考题内容
        for problem in problems:
            problem['content'] = ' '.join(problem['content']).strip()
            problem['full_text'] = '\n'.join(problem['full_text']).strip()
            # 合并标题和内容作为问题
            problem['question'] = f"{problem['title']}\n{problem['content']}" if problem['content'] else problem['title']

        return problems


class AnswerGenerator:
    """答案生成器（从add_answers.py移植并优化）"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processed_count = 0
        self.failed_count = 0
        self.logger = logging.getLogger(__name__)

    def build_prompt(self, question: str, chapter: str) -> str:
        """构建prompt"""
        return f"""
你是一名算法教师，请针对《算法导论》中的题目给出【一种合理的解题思路与分析】，
重点解释算法思想、推导过程与复杂度分析。

注意：
- 不要声称这是"标准答案"或"唯一答案"
- 目标是帮助理解算法思想

章节：{chapter}

题目：
{question}

请按以下结构回答：
1. 问题理解
2. 核心思想 / 算法思路
3. 推导或证明（如需要）
4. 复杂度分析
5. 小结
""".strip()

    def call_llm(self, question: str, chapter: str) -> Optional[str]:
        """调用LLM生成答案"""
        if not self.config.api_key:
            self.logger.warning("未配置API key，跳过答案生成")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": "你是一名资深算法教师。"},
                {"role": "user", "content": self.build_prompt(question, chapter)},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        for attempt in range(1, self.config.retry_times + 1):
            try:
                resp = requests.post(
                    self.config.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout,
                )

                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    self.logger.warning(f"API失败({resp.status_code})，重试 {attempt}/{self.config.retry_times}")

            except Exception as e:
                self.logger.warning(f"请求异常: {e}，重试 {attempt}/{self.config.retry_times}")

            time.sleep(2 ** attempt)

        return None

    def generate_answers(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为所有题目生成答案"""
        # 过滤已生成答案的题目
        pending = [q for q in questions if 'answer' not in q]
        self.logger.info(f"待生成答案的题目数: {len(pending)}")

        for idx, question in enumerate(pending, 1):
            chapter = question.get('chapter', '未知章节')
            question_text = question.get('question', '')

            self.logger.info(f"[{idx}/{len(pending)}] 正在处理: {chapter}")

            answer = self.call_llm(question_text, chapter)

            if answer:
                question['answer'] = answer
                question['answer_meta'] = {
                    "model": self.config.model_name,
                    "generated_at": datetime.now().isoformat()
                }
                self.processed_count += 1
            else:
                question['answer'] = "答案生成失败"
                self.failed_count += 1

            # 限速
            time.sleep(self.config.request_interval)

            # 定期保存checkpoint
            if self.config.save_checkpoint and idx % self.config.checkpoint_interval == 0:
                self.logger.info(f"Checkpoint: 已处理 {idx} 个题目")

        return questions


class QAPipeline:
    """题库生成主Pipeline"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_logging()
        self.extractor = ExerciseExtractor()
        self.answer_generator = AnswerGenerator(config) if config.generate_answers else None

        # 确保输出目录存在
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.join(self.config.output_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                    encoding='utf-8'
                )
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, data: Dict[str, Any], filename: str):
        """保存checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Checkpoint已保存到: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"保存checkpoint失败: {e}")

    def load_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        """加载checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"加载checkpoint失败: {e}")
        return None

    def merge_exercises_and_problems(self, exercises: List[Dict], problems: List[Dict]) -> List[Dict]:
        """合并练习题和思考题为一个列表"""
        all_questions = []

        # 处理练习题
        for ex in exercises:
            # 保留需要的字段
            question_item = {
                'id': ex.get('id', ''),
                'type': 'exercise',
                'chapter': ex.get('chapter', ''),
                'question': ex.get('question', '')
            }
            if 'answer' in ex:
                question_item['answer'] = ex['answer']
                question_item['answer_meta'] = ex.get('answer_meta', {})
            all_questions.append(question_item)

        # 处理思考题
        for prob in problems:
            question_item = {
                'id': prob.get('id', ''),
                'type': 'problem',
                'chapter': prob.get('chapter', ''),
                'question': prob.get('question', '')
            }
            if 'answer' in prob:
                question_item['answer'] = prob['answer']
                question_item['answer_meta'] = prob.get('answer_meta', {})
            all_questions.append(question_item)

        return all_questions

    def generate_summary_csv(self, questions: List[Dict], filename: str):
        """生成题目摘要CSV"""
        import csv

        csv_path = os.path.join(self.config.output_dir, filename)

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['类型', 'ID', '章节', '难度', '题目内容', '行号']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for q in questions:
                writer.writerow({
                    '类型': q['type'],
                    'ID': q['id'],
                    '章节': q['chapter'],
                    '难度': 'normal',  # 默认难度
                    '题目内容': q['question'][:100] + '...' if len(q['question']) > 100 else q['question'],
                    '行号': ''
                })

        self.logger.info(f"摘要CSV已保存到: {csv_path}")

    def run(self) -> Dict[str, Any]:
        """执行完整的pipeline流程"""
        self.logger.info("=== 开始执行题库生成Pipeline ===")

        # 第一步：提取题目
        self.logger.info(f"Step 1: 从 {self.config.input_md_path} 提取题目")
        content = self.extractor.load_file(self.config.input_md_path)

        exercises = self.extractor.extract_exercises(content)
        self.logger.info(f"提取到 {len(exercises)} 道练习题")

        problems = self.extractor.extract_problems(content)
        self.logger.info(f"提取到 {len(problems)} 道思考题")

        # 第二步：生成答案（如果启用）
        if self.config.generate_answers and self.answer_generator:
            self.logger.info("Step 2: 生成题目答案")

            # 尝试加载checkpoint
            checkpoint_data = self.load_checkpoint('checkpoint.json') if self.config.save_checkpoint else None

            if checkpoint_data:
                self.logger.info("发现checkpoint文件，从中恢复进度")
                exercises = checkpoint_data.get('exercises', exercises)
                problems = checkpoint_data.get('problems', problems)

            # 为练习题生成答案
            if exercises:
                self.logger.info("正在为练习题生成答案...")
                exercises = self.answer_generator.generate_answers(exercises)

            # 为思考题生成答案
            if problems:
                self.logger.info("正在为思考题生成答案...")
                problems = self.answer_generator.generate_answers(problems)

            # 保存checkpoint
            if self.config.save_checkpoint:
                checkpoint = {
                    'exercises': exercises,
                    'problems': problems,
                    'timestamp': datetime.now().isoformat()
                }
                self.save_checkpoint(checkpoint, 'checkpoint.json')

        # 第三步：合并并保存结果
        self.logger.info("Step 3: 合并并保存结果")

        # 合并所有题目
        all_questions = self.merge_exercises_and_problems(exercises, problems)

        # 准备最终输出
        result = {
            'questions': all_questions,
            'statistics': {
                'total_exercises': len(exercises),
                'total_problems': len(problems),
                'total_questions': len(all_questions),
                'answered_count': len([q for q in all_questions if 'answer' in q and q['answer'] != '答案生成失败']),
                'generated_at': datetime.now().isoformat()
            },
            'config': asdict(self.config)
        }

        # 保存完整JSON
        output_json = os.path.join(self.config.output_dir, '题库.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.logger.info(f"完整题库已保存到: {output_json}")

        # 保存仅包含答案的精简版
        answered_questions = [q for q in all_questions if 'answer' in q and q['answer'] != '答案生成失败']
        if answered_questions:
            mini_output = os.path.join(self.config.output_dir, '题库_含答案.json')
            with open(mini_output, 'w', encoding='utf-8') as f:
                json.dump({
                    'questions': answered_questions,
                    'statistics': {
                        'total_questions': len(answered_questions),
                        'generated_at': datetime.now().isoformat()
                    }
                }, f, ensure_ascii=False, indent=2)
            self.logger.info(f"含答案题库已保存到: {mini_output}")

        # 生成CSV摘要
        self.generate_summary_csv(all_questions, '题库摘要.csv')

        # 打印统计信息
        stats = result['statistics']
        self.logger.info("\n=== Pipeline完成 ===")
        self.logger.info(f"总题目数: {stats['total_questions']}")
        self.logger.info(f"练习题: {stats['total_exercises']}")
        self.logger.info(f"思考题: {stats['total_problems']}")
        self.logger.info(f"已生成答案: {stats['answered_count']}")

        if self.answer_generator:
            self.logger.info(f"成功生成: {self.answer_generator.processed_count}")
            self.logger.info(f"生成失败: {self.answer_generator.failed_count}")

        return result


def main():
    """主函数 - 命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='算法导论题库生成Pipeline')
    parser.add_argument('input_file', help='输入的markdown文件路径')
    parser.add_argument('-o', '--output', default='./output', help='输出目录（默认: ./output）')
    parser.add_argument('--no-answers', action='store_true', help='不生成答案，仅提取题目')
    parser.add_argument('--api-key', help='API密钥（覆盖配置文件中的设置）')
    parser.add_argument('--config', help='配置文件路径（默认: /home/guoziyang/AIgorithm_Agent/config/base2.yaml）')
    parser.add_argument('--provider', help='指定使用的LLM提供商（覆盖配置文件中的use设置）')

    args = parser.parse_args()

    # 从配置文件加载配置
    config_kwargs = {
        'input_md_path': args.input_file,
        'output_dir': args.output,
        'generate_answers': not args.no_answers,
    }

    # 如果指定了API key，使用命令行的
    if args.api_key:
        config_kwargs['api_key'] = args.api_key
    # 否则尝试从环境变量获取
    elif not os.getenv('OPENAI_API_KEY'):
        # 配置文件中有API key的话会使用配置文件的
        pass

    # 创建配置
    config = PipelineConfig.from_config_file(
        config_file=args.config,
        **config_kwargs
    )

    # 如果指定了provider，从配置文件重新加载对应的配置
    if args.provider and os.path.exists(config.config_file):
        try:
            with open(config.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            llm_config = config_data.get('llm', {})
            provider_config = llm_config.get(args.provider, {})

            if provider_config:
                config.api_key = provider_config.get('api_key', config.api_key)
                config.api_base = provider_config.get('api_base', config.api_base)
                config.model_name = provider_config.get('model', config.model_name)
                config.temperature = provider_config.get('temperature', config.temperature)
                config.top_p = provider_config.get('top_p', config.top_p)
                config.max_tokens = provider_config.get('max_tokens', config.max_tokens)

                # 处理API base URL格式
                if config.api_base and not config.api_base.endswith('/chat/completions'):
                    if 'dmxapi' in config.api_base:
                        config.api_base = config.api_base + '/chat/completions'
                    elif 'siliconflow' in config.api_base:
                        config.api_base = config.api_base + '/chat/completions'
                    elif 'deepseek' in config.api_base:
                        config.api_base = config.api_base + '/chat/completions'

                print(f"使用提供商: {args.provider}")
                print(f"模型: {config.model_name}")
        except Exception as e:
            print(f"警告：切换提供商失败: {e}")

    # 验证输入文件
    if not os.path.exists(config.input_md_path):
        print(f"错误: 输入文件不存在: {config.input_md_path}")
        sys.exit(1)

    # 如果需要生成答案但未提供API key
    if config.generate_answers and not config.api_key:
        print("警告: 未提供API key，将仅提取题目不生成答案")
        config.generate_answers = False

    # 运行Pipeline
    pipeline = QAPipeline(config)
    try:
        result = pipeline.run()
        print("\n✅ Pipeline执行成功！")
    except Exception as e:
        print(f"\n❌ Pipeline执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
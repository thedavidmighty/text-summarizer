from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk
from text_summarizer.entity import ModelEvaluationConfig
import torch
import pandas as pd
from tqdm import tqdm
from evaluate import load

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    

    def generate_batch_sized_bits(self, list_of_items, batch_size):
        for i in range(0, len(list_of_items), batch_size):
            yield list_of_items[i : i + batch_size]

    def calculate_metrics_on_test(self,dataset, metric, model, tokenizer,
                              batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                              column_text="article",
                              column_summary="highlights"):
        # Generate batches
        article_batches = list(self.generate_batch_sized_bits(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_bits(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            # Tokenize the input articles
            inputs = tokenizer(
                article_batch, max_length=1024, truncation=True,
                padding="max_length", return_tensors="pt"
            )
            # # Move inputs to device
            # inputs = {key: val.to(device) for key, val in inputs.items()}

            # Generate summaries
            encoded_summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )

            # Decode generated summaries
            # Replacing the token and the decoded texts with reference to the metrics
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for s in encoded_summaries
            ]
            # Decode target summaries
            decoded_summaries = [d.replace(" ", " ") for d in decoded_summaries]

            # Add predictions and references to the metric
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        # Compute the ROUGE scores
        score = metric.compute()
        return score
    
    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load("rouge")

        score = self.calculate_metrics_on_test(
        dataset_samsum_pt["test"][0:10],
        rouge_metric,
        model_pegasus,
        tokenizer,
        batch_size=2,
        column_text="dialogue",
        column_summary="summary"
        )

        # rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index = ["pegasus"])
        df.to_csv(self.config.metric_file_name, index = False)
    

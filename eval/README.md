# Evaluation Suite

## Quickstart
```bash
bash eval/lighteval_runner.sh serialization_dir/<run_id>
bash eval/harness_runner.sh serialization_dir/<run_id>
python eval/parse_results.py
```

## Benchmarks Included
- MMLU
- GSM8K
- ARC-Challenge
- TruthfulQA
- HellaSwag
- BigBench Hard (BBH)

---

## **How Evaluation Fits into the Project**

1. **After KD training** (RB / FB / RelB), you’ll have saved checkpoints under:
    `serialization_dir/<run_id>/`

2. **Run evals**:
    ```bash
    sbatch eval/lighteval_runner.sh serialization_dir/<run_id>
    sbatch eval/harness_runner.sh serialization_dir/<run_id>
    ```

3. **Parse results**:
    ```bash
    python eval/parse_results.py
    ```

4. **Final aggregated results in**:
    ```bash
    results/eval_summary.csv
    ```



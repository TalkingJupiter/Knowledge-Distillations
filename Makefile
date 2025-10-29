.PHONY: submit_rb submit_fb submit_relb submit_all cancel clean_logs clean_checkpoints

submit_rb:
	sbatch scripts/distil/Qwen/Qwen2.5-1.5B-Instruct/run_response_based_single_node.sh

submit_fb:
	sbatch scripts/distil/Qwen/Qwen2.5-1.5B-Instruct/run_feature_based_single_node.sh

submit_relb:
	sbatch scripts/distil/Qwen/Qwen2.5-1.5B-Instruct/run_relation_based_single_node.sh

submit_all: submit_rb submit_fb submit_relb
	@echo "✅ Submitted all jobs!"

cancel:
	scancel --me

clean_logs:
	rm -rf logs/Qwen/2.5-1.5B-Instruct/*
	rm -rf telemetry/Qwen/2.5-1.5B-Instruct/*
	
clean_checkpoints:
	rm -rf serialization_dir/Qwen/2.5-1.5B-Instruct/*

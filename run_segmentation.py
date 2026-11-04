import argparse, os, sys
import batch_run_inference_hn_oars as batch

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['CT_HeadAndNeck_OARs'])
parser.add_argument('--data_dir', required=True)
parser.add_argument('--results_dir', required=True)
parser.add_argument('--session_dir', default=None)
args = parser.parse_args()
sessionDir = args.session_dir or os.path.join(args.results_dir, 'session')

# run_inference_selfattn calls TrainOptions().parse(), which parses sys.argv and has its own --model
sys.argv = sys.argv[:1]

errorLog = os.path.join(args.results_dir, 'errors.log')
if os.path.exists(errorLog):
    os.remove(errorLog)
batch.main(args.data_dir, sessionDir, args.results_dir)
if os.path.exists(errorLog):
    sys.exit(f"See {errorLog} for failed cases")

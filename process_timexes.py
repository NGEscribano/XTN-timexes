import argparse
import sys
from argparse import RawDescriptionHelpFormatter

"""
PROCESSOR FOR XTN-TIMEXES

This program processes timexes for the XTN multilingual timex processing system.
It performs two different steps:

1. Extraction of timexes predicted by the detection model to feed the TimeNorm normalizer. It requires:
	--gold_file [GOLD_FILE] File with gold detection and normalization data in BIO format.
	--det_file [DET_FILE] Output file from the timex detection model in BIO format.
	--out_file [OUT_FILE] Output file to feed the TimeNorm system with predicted timexes and gold normalization data.

2. Combination of output data from the detection model and the TimeNorm system.
It requires:
	--gold_file [GOLD_FILE] File with gold detection and normalization data in BIO format.
	--det_file [DET_FILE] Output file from the timex detection model in BIO format.
	--norm_file [NORM_FILE] Output file from TimeNorm.
	--out_file [OUT_FILE] Output file with all the gold and predicted data from detection and normalization.
"""


def main(args):
    # Obtain gold data from the gold file and detection-prediction data from the
    # model output file
    with open(args.gold_file, "r") as gold, open(args.det_file, "r") as det:
        gold_data = gold.readlines()
        det_data = det.readlines()

    # Check if gold and det files have the same length
    if len(gold_data) != len(det_data):
        sys.exit("ERROR: Gold file and det file do not have the same length.")

    tokens = get_column(gold_data, 0)
    gold_tags = get_column(gold_data, 1)
    gold_vals = get_column(gold_data, -1)
    pred_tags = get_column(det_data, -1)

    # PREPARE DETECTED TIMEX SPANS FOR NORMALIZATION

    # If norm_file is not provided, prepare detected timex spans for normalization
    if not args.norm_file:
        # Obtain predicted spans and their gold normalizations
        timexes = get_pred_spans(tokens, gold_tags, pred_tags, gold_vals)

        # Write the timex spans and values
        with open(args.out_file, "w") as output:
            for i, line in enumerate(timexes):
                # If the line is an empty string, write a newline to mark new DCT
                if line == "":
                    if i != 0:
                        output.write("\n")
                # If the line is a timex with no value, write only the expression
                elif line[1] == "":
                    output.write(f"{line[0]}\n")
                # If the line is a timex with a value, write them both
                else:
                    output.write(f"{line[0]}\t{line[-1]}\n")
            print("Output file created. Feed TimeNorm with this output file.")


    # JOIN DET AND NORM RESULTS IN BIO FORMAT

    # If norm_file is provided, join all the gold and prediction data from detection and normalization
    else:
        with open(args.norm_file, "r") as norm:
            norm_data = norm.readlines()
        timexes = get_column(norm_data, 0)
        norms = get_column(norm_data, 2)

        norm2bio(tokens, gold_tags, pred_tags, gold_vals, timexes, norms, args.out_file)
        print("Output file created. Evaluate detection and normalization results in this file with the evaluator.")


def get_column(file_data, col_n):
    """Obtain a concrete column of a file as a list."""
    # Check if the separator is a tab or a whitespace
    sep = "\t" if "\t" in file_data[0] else " "
    # If line is not empty, append the corresponding tag
    column = []
    for line in file_data:
        if line != "\n":
            column.append(line.strip().split(sep)[col_n])
        else:
            column.append("")
    return column


def split_tag(chunk_tag):
    """Splits chunk tags into IOBES prefix and chunk type."""
    # If chunk tag is "O" or an empty line, return ("O", None)
    if chunk_tag == "O" or chunk_tag == "":
        return "O", None
    # Otherwise, return the corresponding position and type
    return chunk_tag.split("-", maxsplit=1)


def is_chunk_end(prev_tag, tag):
    """Checks if a chunk ended between the previous and current word."""
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == "O":
        return False
    elif prefix2 == "B":
        return True
    elif prefix2 == "O":
        return prefix1 != "O"
    else:
        return False


def get_pred_spans(tokens, gold_tags, pred_tags, norm_tags):
    """It takes all gathered data and returns the list of predicted timex spans
    along with their gold normalization values to feed the normalizer."""
    # Lists for final timexes and temporal pred tokens
    timexes, pred_tokens = [], []
    # Initialize variables
    prev_pred_tag, value, last_pred_type, counter = "O", None, None, 0
    # Gather all data in a zipped list
    data = list(zip(tokens, gold_tags, pred_tags, norm_tags))

    for i, (token, gold_tag, pred_tag, norm_tag) in enumerate(data):
        # Obtain predicted token type position and tag
        pred_pos, pred_type = split_tag(pred_tag)
        # Check if this token is the end of a timex
        pred_end = is_chunk_end(prev_pred_tag, pred_tag)

        # If the token represents a DCT, append an empty string and the value
        if token.startswith("*!-DCTVALUE="):
            if i != 1:
                timexes.append("")
            dct = token.replace('*!-DCTVALUE="', '')
            dct = dct.replace('"-!*', '')
            timexes.append((dct, dct))

        # If a timex has ended, store the expression and value
        if pred_end:
            timexes.append((" ".join(pred_tokens), value))
            pred_tokens.clear()
            value = None
            counter += 1

        # If the token is part of a timex, append it and set its value
        if pred_pos != "O":
            value = norm_tag
            pred_tokens.append(token)

        # If the last token is a timex, store its data
        if i == len(data) - 1 and value is not None:
            timexes.append((" ".join(pred_tokens), value))
            counter += 1

        # Turn current tags into previous tags
        prev_pred_tag = pred_tag

    print("Detected timex spans (including DCTs):", counter)

    return timexes


def norm2bio(tokens, gold_tags, pred_tags, gold_vals, timexes, norms, out_file):
    """Joins detection and normalization gold and pred data."""
    # Turn timex data into "token norm" format, ignoring DCTs
    timex_tokens, pred_vals, new_dct = [], [], True
    for i, timex in enumerate(timexes):
        if timex != "":
            if new_dct:
                new_dct = False
            else:
                for timex_token in timex.split(" "):
                    timex_tokens.append(timex_token)
                    pred_vals.append(norms[i])
        else:
            new_dct = True

    # Obtain predicted value for each predicted timex token
    det_data = list(zip(tokens, pred_tags))
    pred_norm_tags = []
    for i, (token, pred_tag) in enumerate(det_data):
        if token != "":
            if pred_tag != "O" and token == timex_tokens[0]:
                pred_norm_tags.append(pred_vals[0])
                timex_tokens.pop(0), pred_vals.pop(0)
            else:
                pred_norm_tags.append("-")
        else:
            pred_norm_tags.append("")

    # Gather and print all the det and norm data
    all_data = zip(tokens, gold_tags, pred_tags, gold_vals, pred_norm_tags)
    with open(out_file, "w") as output:
        for token, gold_tag, pred_tag, gold_val, pred_norm_tag in all_data:
            if token != "":
                output.write(f"{token}\t{gold_tag}\t{pred_tag}\t{gold_val}\t{pred_norm_tag}\n")
            else:
                output.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
	This program processes timexes for the XTN multilingual timex processing system.\n
	It performs two steps:\n\n

	1. Extraction of timexes predicted by the detection model to feed the TimeNorm normalizer. It requires:\n
		--gold_file [GOLD_FILE] File with gold detection and normalization data in BIO format.\n
		--det_file [DET_FILE] Output file from the timex detection model in BIO format.\n
		--out_file [OUT_FILE] Output file to feed the TimeNorm system with predicted timexes and gold normalization data.\n\n

	2. Combination of output data from the detection model and the TimeNorm system. It requires:\n
		--gold_file [GOLD_FILE] File with gold detection and normalization data in BIO format.\n
		--det_file [DET_FILE] Output file from the timex detection model in BIO format.\n
		--norm_file [NORM_FILE] Output file from TimeNorm.\n
		--out_file [OUT_FILE] Output file with all the gold and predicted data from detection and normalization.\n
	\n""",
    formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("-g", "--gold_file", required=True,
        help="To extract detected timex spans to feed the normalizer. "
        "Contains gold detection and normalization data about each token in '{token}\t{gold-det}\t{gold-norm}' format. "
        "Normalization value must be indicated for all tokens of each timex.")
    parser.add_argument("-d", "--det_file", required=True,
        help="Contains gold and detection data in '{token}\t[{gold-det}\t]{pred-det}' format.")
    parser.add_argument("-n", "--norm_file", required=False,
        help="Contains gold and normalization data in '{timex}\t{gold-norm}\t{pred-norm}' format.")
    parser.add_argument("-o", "--out_file", required=True,
        help="Outputs data from one of the two steps processed by this program.")

    args = parser.parse_args()
    main(args)

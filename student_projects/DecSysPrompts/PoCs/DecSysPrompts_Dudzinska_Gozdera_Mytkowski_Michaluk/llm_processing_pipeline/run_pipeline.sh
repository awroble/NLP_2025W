DATA_DIR=../CL4R1T4S-no-code

python extract_institutional_statements.py --system-prompt-type parsing --root-dir $DATA_DIR --output-dir-suffix=json-statements
python extract_institutional_statements.py --system-prompt-type classification --root-dir ${DATA_DIR}-json-statements --output-dir-suffix=classified

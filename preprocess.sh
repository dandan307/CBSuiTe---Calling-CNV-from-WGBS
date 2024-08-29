mode="germline"

if [ "$mode" == "germline" ]; then
    python -u scripts/preprocess_sample_call_germline.py -h
elif [ "$mode" == "somatic" ]; then
    python -u scripts/preprocess_sample_call_somatic.py -h
else
    echo "Invalid mode specified. Please use 'germline' or 'somatic'."
fi


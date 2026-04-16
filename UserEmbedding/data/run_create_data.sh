# org_data_folder="/raid/ltnghia02/vyttt/dance_gen/UserEmbedding/datasets/edge_aistpp"
# train_data_folder="/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp_processed/train"
# test_data_folder="/raid/ltnghia02/vyttt/catb/dance_gen/UserEmbedding/datasets/edge_aistpp_processed/test"

# train_data_file="/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/train.txt"
# test_data_file="/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/test.txt"
# ignore_data_file="/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/ignore_list.txt"

# shuffle_file="/raid/ltnghia02/vyttt/catb_code/dance_gen/UserEmbedding/data/splits/shuffle.txt"

# test run
org_data_folder="./UserEmbedding/data/edge_aistpp"
train_data_folder="./UserEmbedding/data/train"
test_data_folder="./UserEmbedding/data/test"

train_data_file="./UserEmbedding/data/splits/train.txt"
test_data_file="./UserEmbedding/data/splits/test.txt"
ignore_data_file="./UserEmbedding/data/splits/ignore_list.txt"

shuffle_file="./UserEmbedding/data/splits/shuffle.txt"

start_time=$(date +%s)

python ./UserEmbedding/data/create_dataset.py \
    --dataset_folder "$org_data_folder" \
    --train_folder "$train_data_folder" \
    --test_folder "$test_data_folder" \
    --train_data_file "$train_data_file" \
    --test_data_file "$test_data_file" \
    --ignore_data_file "$ignore_data_file" \
    --do_shuffle \
    --shuffling_map_file "$shuffle_file"
    
# python create_dataset.py \
#     --dataset_folder "$org_data_folder" \
#     --train_folder "$train_data_folder" \
#     --test_folder "$test_data_folder" \
#     --train_data_file "$train_data_file" \
#     --test_data_file "$test_data_file" \
#     --ignore_data_file "$ignore_data_file" \

elapsed=$(( $(date +%s) - start_time ))
printf 'Elapsed time: %02d:%02d:%02d\n' \
  $((elapsed/3600)) \
  $(((elapsed%3600)/60)) \
  $((elapsed%60))

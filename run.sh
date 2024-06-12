python run.py --log_name HREC_best --method 1 --batch_size 256
# python run.py --log_name DNN_time --method 2 --batch_size 2
# python run.py --log_name Transformer_CNN_time --method 4 --batch_size 2
# python run.py --log_name BIGRU_Attention_time --method 3 --batch_size 2


# python run.py --log_name HREC_best --method 1 --batch_size 256 --is_drop true
# # python run.py --log_name DNN_random_drop --method 2 --batch_size 256 --is_drop true
# # python run.py --log_name Transformer_CNN_random_drop --method 4 --batch_size 256 --is_drop true
# # python run.py --log_name BIGRU_Attention_random_drop --method 3 --batch_size 256 --is_drop true

# token="ALL" # ALL parameters source_code ast node_type   
# token=("parameters" "source_code" "ast" "node_type")
# echo ${token}
# for i in "${!token[@]}"
# do
#     token_type=${token[$i]}
#     python run.py --log_name HREC_${token_type}_best --method 1 --batch_size 256 --token_type ${token_type} 
#     # python run.py --log_name DNN_${token_type} --method 2 --batch_size 256 --token_type ${token_type} 
#     # python run.py --log_name Transformer_CNN_${token_type} --method 4 --batch_size 256 --token_type ${token_type} 
#     # python run.py --log_name BIGRU_Attention_${token_type} --method 3 --batch_size 256 --token_type ${token_type} 
# done

# cp ./logs/* ./log_ed/
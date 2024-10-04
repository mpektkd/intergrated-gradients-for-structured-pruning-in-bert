!/bin/bash

# Remote server information
# REMOTE_USER="dbekris"
# REMOTE_HOST="147.102.3.211"
# REMOTE_BASE_DIR="/home/dbekris/src/scores"

# # Local directory
# LOCAL_BASE_DIR="C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores"

# # Password
# PASSWORD="giemoupoupasmana8apawstakaravia"

# # Datasets and M values
# DATASETS=('imdb' 'sst2' 'rotten_tomatoes')
# M_VALUES=('None' '0' '1' 'class_reduce')

# # Loop through datasets and M values
# for dataset in "${DATASETS[@]}"; do
#     for m in "${M_VALUES[@]}"; do
#         LOCAL_DIR="${LOCAL_BASE_DIR}/${dataset}/${m}Class/200tokens/fine_tuned"
#         # if [ ! -d "${LOCAL_DIR}" ]; then
#         #     mkdir -p "${LOCAL_DIR}"
#         # fi
#         REMOTE_DIR="${REMOTE_BASE_DIR}/${dataset}/${m}Class/200tokens/fine_tuned"

#         # echo ${LOCAL_DIR}
#         # echo ${REMOTE_DIR}
#         # Transfer files
#         C:\\pscp -pw ${PASSWORD} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/corr.pt ${LOCAL_DIR}
#         C:\\pscp -pw ${PASSWORD} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mutual.pt ${LOCAL_DIR}
#         C:\\pscp -pw ${PASSWORD} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/kl_div.pt ${LOCAL_DIR}
#     done
# done

# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/NoneClass/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/NoneClass/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/NoneClass/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/NoneClass/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/0Class/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/0Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/0Class/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/0Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/1Class/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/1Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/1Class/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/1Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/NoneClass/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/NoneClass/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/NoneClass/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/NoneClass/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/0Class/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/0Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/0Class/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/0Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/1Class/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/1Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/1Class/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/1Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned/stats.txt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned
# C:\\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned/visualization.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned



# -------------------------------------------------
# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/NoneClass/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/NoneClass/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/NoneClass/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/NoneClass/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/0Class/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/0Class/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/0Class/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/0Class/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/1Class/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/1Class/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/1Class/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/1Class/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/class_reduce/150tokens/pre_trained/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/class_reduce/150tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/class_reduce/150tokens/pre_trained/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/class_reduce/150tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/class_reduce/150tokens/pre_trained/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/class_reduce/150tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/imdb/class_reduce/150tokens/pre_trained/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/imdb/class_reduce/150tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/NoneClass/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/NoneClass/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/NoneClass/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/NoneClass/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/0Class/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/0Class/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/0Class/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/0Class/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/1Class/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/1Class/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/1Class/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/1Class/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/class_reduce/200tokens/pre_trained/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/class_reduce/200tokens/pre_trained/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/class_reduce/200tokens/pre_trained/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/sst2/class_reduce/200tokens/pre_trained/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/sst2/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/NoneClass/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/0Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/1Class/200tokens/fine_tuned

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained/corr.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained/mutual.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained/kl_div.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained

# C:\pscp -pw giemoupoupasmana8apawstakaravia dbekris@147.102.3.211:/home/dbekris/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained/kl_div_rev.pt C:/Users/mpek/Desktop/hmmy_ntua/Thesis/thesis-layerconductance-structured-pruning-bert/src/scores/rotten_tomatoes/class_reduce/200tokens/pre_trained
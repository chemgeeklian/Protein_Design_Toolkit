def parse_clstr(file_path):
    '''
    parse .clust file from cd-hit to make a python dictionary.
    
    cd-hit clusters proteins into clusters that meet a user-defined similarity threshold, 
    usually a sequence identity. Each cluster has one representative sequence. 
    The input is a protein dataset in fasta format and the output are two files: 
    a fasta file of representative sequences and a text file of list of clusters.
    '''
  
    with open(file_path, 'r') as file:
        cluster_data = file.read()

    lines = cluster_data.split("\n")

    clusters = {}
    current_cluster = None
    for line in lines:
        line = line.strip()  # remove leading/trailing whitespace
        if line:  # check if line is not blank
            if line.startswith('>Cluster'):
                # We've started a new cluster
                current_cluster = int(line.split(' ')[1])
                clusters[current_cluster] = []
            else:
                # Extract the uniprot id from the line
                try:
                    uniprot_id = line.split(", >")[1].split("...")[0]
                    clusters[current_cluster].append(uniprot_id)
                except IndexError:
                    print(f"Couldn't parse line: {line}")
                  
    return clusters

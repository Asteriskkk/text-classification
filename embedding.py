def load_vectors(fname):
    # Download GloVe vectors (uncomment the below)
    GLOVE_FILENAME = fname
    glove_index = {}
    n_lines = sum(1 for line in open(GLOVE_FILENAME, encoding="utf8"))
    with open(GLOVE_FILENAME, encoding="utf8") as fp:
        for line in tqdm(fp, total=n_lines):
            split = line.split()
            word = split[0]
            vector = np.array(split[1:]).astype(float).tolist()
            glove_index[word] = vector
    print("total length index",len(glove_index))
    return glove_index
  










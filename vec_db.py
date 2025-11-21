from typing import Annotated
import numpy as np
import os
import struct
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
TOP_K_DEFAULT = 5

# IVF clusters choice
IVF_CONFIGS = {
    1_000_000: 4096,     # higher cluster resolution for 1M
    10_000_000: 8192,    # balanced recall/time
    15_000_000: 12288,   # large-scale precision
    20_000_000: 16384    # fine-grained indexing for 20M
}

# PQ M choice
# PQ_M_CONFIGS = {
#     1_000_000: 16,       # 70/16 ≈ 4D subvector
#     10_000_000: 16,
#     15_000_000: 16,
#     20_000_000: 32       # more precision for largest DB
# }
## de mesh mofeda zeyadetha asl wana batrain batrain 3ala 500000 kda kda fa mesh hyfe2 we wana baretrieve be retreive 3ala probe_cluster fe 3add el el data ele gowa kol cluster
# ele howa 8aleban bardo 3add sabet 34an 3add el ivf clusters byzed ma3a el size
# Number of clusters to probe
PROBE_CLUSTERS = 60

# Chunking and sampling for build phase
BUILD_CHUNK = 10_000
SAMPLE_SIZE = 500_000


class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat",
                 new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            self.db_size = db_size

            # remove old files if present
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self._cleanup_index_files()
            self.generate_database(db_size)

    def _cleanup_index_files(self):
        # remove possible index files
        base = self.index_path.replace(".dat", "")
        for f in os.listdir("."):
            if f.startswith(base):
                try:
                    os.remove(f)
                except:
                    pass

    # --------------------------- DB CREATION --------------------------- #
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        # build index from disk (no caching)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        # rebuild index (simple approach)
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"get_one_row error: {e}")
            return None

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    # --------------------------- INDEX BUILD: IVF + PQ --------------------------- #
    def _choose_n_clusters(self):
        # pick nearest config
        closest = min(IVF_CONFIGS.keys(), key=lambda x: abs(x - self.db_size))
        return IVF_CONFIGS[closest]

    def _choose_pq_m(self):
        # closest = min(PQ_M_CONFIGS.keys(), key=lambda x: abs(x - self.db_size))
        # return PQ_M_CONFIGS[closest]
        return 32

    def _build_index(self):
        """
        Build IVF + PQ index from on-disk vectors.
        - Centroids saved to index_path_centroids.npy
        - PQ codebooks saved to index_path_pq.npy (shape: list of arrays or stacked)
        - Per-cluster inverted lists saved to index_path_cluster_{cid}.bin
          Each entry in cluster file: struct of M uint8 (code) followed by uint32 (vector id)
        - cluster_counts saved to index_path_cluster_counts.npy
        """
        num_records = self._get_num_records()
        if num_records == 0:
            raise RuntimeError("No vectors to index")

        n_clusters = self._choose_n_clusters()
        M = self._choose_pq_m()
        # ensure M divides DIMENSION or handle last subspace
        subdims = []
        base = DIMENSION // M
        rem = DIMENSION % M
        for i in range(M):
            subdims.append(base + (1 if i < rem else 0))

        print(f"Building IVF+PQ: N={num_records}, clusters={n_clusters}, PQ M={M}, subdims={subdims}")

        # -------------------- step 1: sample data --------------------
        samp = min(SAMPLE_SIZE, num_records)
        rng = np.random.default_rng(DB_SEED_NUMBER)
        sample_idxs = rng.choice(num_records, size=samp, replace=False)
        # load sample
        mm = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        sample = np.array(mm[sample_idxs])  # fits in memory (SAMPLE_SIZE x DIM)
        # normalize sample rows (so PQ approximates cosine reasonably)
        sample_norms = np.linalg.norm(sample, axis=1, keepdims=True)
        sample_norms[sample_norms == 0] = 1.0
        sample = sample / sample_norms

        '''
        ana hena ba5tar random sample_size rows we ba3mel le kol wa7ed normalization ele howa
        ba2sem 3al length aw el magnitude 34an wana bamatch fel pq a2der a3mel dot product 3ala tol

        '''

        # -------------------- step 2: train IVF centroids --------------------
        print("Training MiniBatchKMeans for centroids...")
        kmeans_ivf = MiniBatchKMeans(n_clusters=n_clusters, random_state=DB_SEED_NUMBER, batch_size=4096, n_init="auto")
        kmeans_ivf.fit(sample)
        centroids = kmeans_ivf.cluster_centers_.astype(np.float32)
        # normalize centroids
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1.0
        centroids = centroids / c_norms
        np.save(self.index_path.replace(".dat", "_centroids.npy"), centroids)
        print("Centroids saved.")

        # -------------------- step 3: train PQ codebooks (per subspace) --------------------
        print("Training PQ codebooks...")
        pq_codebooks = []
        start_idx = 0
        for sd in subdims:
            subspace = sample[:, start_idx:start_idx + sd]
            # print(subspace.shape)
            # cluster into 256 centroids
            kmeans_pq = MiniBatchKMeans(n_clusters=256, random_state=DB_SEED_NUMBER, batch_size=4096, n_init="auto")
            kmeans_pq.fit(subspace)
            codebook = kmeans_pq.cluster_centers_.astype(np.float32)
            # no extra normalization here; keeping raw codebooks
            pq_codebooks.append(codebook)
            start_idx += sd
        # Save PQ codebooks as a single .npz for convenience
        base = self.index_path.replace(".dat", "")
        np.savez(base + "_pq_codebooks.npz", *pq_codebooks)
        print("PQ codebooks saved.")#(265,32)

        # -------------------- step 4: create empty cluster files and counts --------------------
        cluster_counts = np.zeros(n_clusters, dtype=np.int64)
        cluster_paths = [f"{base}_cluster_{i}.bin" for i in range(n_clusters)]
        # create (truncate) files
        for p in cluster_paths:
            open(p, "wb").close()

        # -------------------- step 5: assign vectors in chunks and encode PQ codes --------------------
        print("Assigning and encoding vectors in chunks...")
        mm_all = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        for start in range(0, num_records, BUILD_CHUNK):
            end = min(start + BUILD_CHUNK, num_records)
            chunk = np.array(mm_all[start:end], dtype=np.float32)  # small chunk in memory
            # normalize chunk
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            chunk_n = chunk / norms  # normalized for centroid assignment & PQ encoding

            # compute similarity to centroids (dot product because normalized)
            sims = chunk_n @ centroids.T  # shape (chunk_size, n_clusters)
            nearest = np.argmax(sims, axis=1)  # cluster id for each vector in chunk

            # encode PQ codes for chunk (vectorized)
            codes_chunk = np.empty((end - start, M), dtype=np.uint8)
            s = 0
            for m_idx, sd in enumerate(subdims):
                subvecs = chunk_n[:, s:s + sd]  # shape (chunk_size, sd)
                codebook = pq_codebooks[m_idx]  # shape (256, sd)
                # compute distances to codebook centroids: use squared L2
                # For vectorized efficiency: use (a-b)^2 = a^2 + b^2 - 2a.b
                a2 = np.sum(subvecs ** 2, axis=1, keepdims=True)  # (chunk,1)
                b2 = np.sum(codebook ** 2, axis=1)  # (256,)
                dots = subvecs @ codebook.T  # (chunk,256)
                # squared distances:
                dists = a2 + b2 - 2.0 * dots  # (chunk,256)
                nearest_codes = np.argmin(dists, axis=1).astype(np.uint8)
                codes_chunk[:, m_idx] = nearest_codes
                s += sd

            # group bytes to write per cluster to minimize file calls
            buffers: Dict[int, bytearray] = {}
            counts_local: Dict[int, int] = {}

            for local_idx, cid in enumerate(nearest):
                global_id = start + local_idx
                code_bytes = codes_chunk[local_idx].tobytes()  # M bytes
                id_bytes = struct.pack('<I', int(global_id))  # 4 bytes, little-endian unsigned int
                if cid not in buffers:
                    buffers[cid] = bytearray()
                    counts_local[cid] = 0
                buffers[cid].extend(code_bytes)
                buffers[cid].extend(id_bytes)
                counts_local[cid] += 1

            # write buffers to files (append)
            for cid, buff in buffers.items():
                with open(cluster_paths[cid], "ab") as fh:
                    fh.write(buff)
                cluster_counts[cid] += counts_local[cid]

        # Save counts
        np.save(base + "_cluster_counts.npy", cluster_counts)
        print("Index build complete. Cluster counts saved.")

    # --------------------------- RETRIEVE (NO CACHING) --------------------------- #
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=TOP_K_DEFAULT):
        """
        Disk-only retrieval using IVF+PQ + ADC.
        No caching: loads centroids & PQ codebooks per call (tiny), reads cluster files sequentially.
        """
        if query.shape != (1, DIMENSION):
            query = query.reshape(1, -1)
        q = query.astype(np.float32).copy().reshape(-1)
        # normalize query
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            qnorm = 1.0
        q = q / qnorm

        base = self.index_path.replace(".dat", "")

        if not (os.path.exists(base + "_centroids.npy") and
          os.path.exists(base + "_pq_codebooks.npz") and
          os.path.exists(base + "_cluster_counts.npy")):
          print("Index files not found. Building index automatically...")

    # Try to infer DB size safely if not set
          if not hasattr(self, "db_size") or self.db_size is None:
                try:
                  self.db_size = self._get_num_records()
                  print(f"Inferred db_size = {self.db_size}")
                except Exception as e:
                  raise RuntimeError(f"Cannot infer DB size before building index: {e}")

    # Build the index now
          self._build_index()
        # Load centroids (tiny) and pq codebooks (tiny) and counts
        centroids = np.load(base + "_centroids.npy")  # (n_clusters, DIM)
        pq_npz = np.load(base + "_pq_codebooks.npz")
        pq_codebooks = [pq_npz[key] for key in pq_npz.files]  # list of (256, subdim)
        cluster_counts = np.load(base + "_cluster_counts.npy")
        n_clusters = centroids.shape[0]
        M = len(pq_codebooks)

        # compute similarity to centroids and pick top PROBE_CLUSTERS
        sims = centroids @ q  # (n_clusters,)
        top_clusters = np.argsort(sims)[-PROBE_CLUSTERS:][::-1]

        # Precompute ADC tables T[m, 256]: distances between query subvector and PQ centroids
        # define subdims same as during build
        # compute subdims from pq_codebooks shapes
        subdims = [cb.shape[1] for cb in pq_codebooks]
        # split query into subspaces
        q_subs = []
        s = 0
        for sd in subdims:
            q_subs.append(q[s:s + sd])
            s += sd

        # compute T
        T = []
        for m_idx in range(M):
            codebook = pq_codebooks[m_idx]  # (256, sd)
            # compute squared L2 distances between q_subs[m] and all 256 centroids
            a2 = np.sum(q_subs[m_idx] ** 2)
            b2 = np.sum(codebook ** 2, axis=1)  # (256,)
            dots = codebook @ q_subs[m_idx]  # (256,)
            dists = a2 + b2 - 2.0 * dots  # (256,)
            T.append(dists.astype(np.float32))
        T = np.array(T)  # shape (M, 256)

        # Gather candidates
        candidates_ids = []
        candidates_scores = []  # lower = closer in L2; we'll pick smallest distances

        # For each probed cluster, read its file and compute ADC distances vectorized
        for cid in top_clusters:
            cluster_path = f"{base}_cluster_{cid}.bin"
            if not os.path.exists(cluster_path):
                continue
            count = int(cluster_counts[cid])
            if count == 0:
                continue
            # read full content (sequential read)
            with open(cluster_path, "rb") as fh:
                raw = fh.read()
            # interpret as structured array: codes (M x uint8) + id (uint32)
            dtype = np.dtype([('code', np.uint8, M), ('id', np.uint32)])
            try:
                arr = np.frombuffer(raw, dtype=dtype)
            except Exception as e:
                # in case of alignment / corruption, skip
                continue
            if arr.size == 0:
                continue
            codes = arr['code']  # shape (count, M), dtype uint8
            ids = arr['id'].astype(np.int64)

            # ADC: for each candidate, sum T[m, code[m]] across m
            # Vectorized: build (count, M) from indexing
            # T is (M, 256). For each m, T[m, codes[:, m]] -> (count,)
            dist_sum = np.zeros(codes.shape[0], dtype=np.float32)
            for m_idx in range(M):
                dist_sum += T[m_idx][codes[:, m_idx]]

            candidates_ids.append(ids)
            candidates_scores.append(dist_sum)

        if len(candidates_ids) == 0:
            return []

        # concatenate arrays
        all_ids = np.concatenate(candidates_ids)
        all_scores = np.concatenate(candidates_scores)

        # get top_k smallest distances
        if all_scores.size <= top_k:
            top_idx = np.argsort(all_scores)
        else:
            top_idx = np.argpartition(all_scores, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(all_scores[top_idx])]

        top_ids = all_ids[top_idx].tolist()
        return top_ids

    # --------------------------- UTILITIES --------------------------- #
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return -1.0
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return float(cosine_similarity)

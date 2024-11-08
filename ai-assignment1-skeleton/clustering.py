import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# argparse를 사용하여 명령줄에서 인자를 받을 수 있도록 설정
parser = argparse.ArgumentParser(description='AI course, CAU')
parser.add_argument('--data_type', type=int, default=0, help='0: Gaussian data, 1: Moon-shaped data')
parser.add_argument('--cluster_method', type=int, default=0, help='0: k-Means, 1: GMM, 2: DBSCAN')
parser.add_argument('--cluster_num', type=int, default=3, help='The number of clusters')
parser.add_argument('--eps', type=int, default=0.2, help='The epsilon for DBSCAN')
parser.add_argument('--min_pts', type=int, default=5, help='The minimum number of data points')

# 명령줄 인자를 파싱
args = parser.parse_args()


def generate_data(type):
    if type == 0:
        # 가우시안 분포 데이터를 생성
        # 각 클러스터에 대해 100개의 데이터를 샘플링
        sample_sizes = [100, 100, 100]

        # 각 클러스터의 중심, µ1 = [7.0, −1.0], µ2 = [3.0, −1.5], µ3 = [5.5, 1.0]
        means = [[7.0, -1.0], [3.0, -1.5], [5.5, 1.0]]

        # 표준 편차 is same for all clusters σ = 0.7
        stds = [0.7, 0.7, 0.7]

        # 가우시안 분포에 따른 데이터를 생성
        # Data type 1: double moon shaped data
        X, y = make_blobs(n_samples = sample_sizes, centers = means, n_features = 2, cluster_std = stds, random_state=0)
        print (X)

        # 생성된 데이터 시각화
        plt.scatter(X[:, 0], X[:, 1], s=50, cmap='Paired')
        #plt.show()

        return X, y

    else:
        # 더블 문(half-moon) 모양의 데이터를 생성
        # Data type 2: double moon shaped data
        X, y = make_moons(200, noise=.05, random_state=0)
        print (X)

        # 생성된 데이터 시각화
        plt.scatter(X[:, 0], X[:, 1], s=50, cmap='Paired')
        #plt.show()

        return X, y



def kmeans(X, max_iterations, k):
    # 입력 파라미터 설명
    # X: 학습 데이터 포인트
    # max_iterations: 최대 반복 횟수
    # k: 클러스터의 개수

    # (TODO) Write your code here
    dataSize = X.shape[0]
    memberships = np.zeros(dataSize, dtype=int)

    # 1.각 cluster의 중심을 data point 집합 D 내의 임의의 위치로 초기화 (cluster 중심 랜덤 초기화)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    count = 0
    epsilon = 1e-4
    centro_shift = 100

    # 2~3을 cluster의 중심점이 더이상 변하지 않거나, 최대 반복횟수에 도달할 때까지 반복
    while count < max_iterations and centro_shift > epsilon:
        previous_centroids = np.copy(centroids)
        # 2. 각 데이터 포인트 x_i에 대해, 가장 가까운 cluster 찾기. (x_i가 어디에 속하나)
        for i in range(dataSize):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            memberships[i] = np.argmin(distances)
            # closest_cluster = np.argmin(distances)
            # memberships[i, closest_cluster] = 1 # 가장 가까운 cluster에 1 할당

        # 3. cluster 중심 업데이트
        for j in range(k):
            #points_in_cluster = X[memberships[:, j] == 1] # cluster j에 속하는 x들의 인덱스 선택
            points_in_cluster = X[memberships == j]  # 클러스터 j에 속하는 포인트 선택
            if len(points_in_cluster) > 0: # cluster에 속한 x가 존재하는 경우
                centroids[j] = points_in_cluster.mean(axis=0)  # cluster 중심 업데이트
        
        centro_shift = np.linalg.norm(centroids - previous_centroids)
        count +=1
    
    # 반환 파라미터
    # centroids: 클러스터 중심점
    # memberships: a list of memberships for data points(각 데이터 포인트의 클러스터 할당)
    return centroids, memberships


def gmm(X, max_iterations, k):
    # 입력 파라미터 설명
    # X: 학습 데이터 포인트
    # max_iterations: 최대 반복 횟수
    # k: 클러스터의 개수

    tolerance = 1e-6  # 파라미터 변화의 허용 오차
    iteration = 0
    converged = False   

    # parameters : 평균, 공분산, cluster 평균 벡터

    dataSize = X.shape[0] # 데이터 포인트 개수
    mus = np.random.rand(k, X.shape[1])  # 각 클러스터의 평균 벡터 (임의 값으로 초기화)
    covs = np.array([np.eye(X.shape[1]) for _ in range(k)])  # 각 클러스터의 공분산 행렬 (단위 행렬로 초기화)
    pis = np.full(k, 1/k)  # 각 클러스터의 prior 확률 (모두 동일한 확률로 초기화)

    memberships = np.zeros((dataSize, k)) # Posterior 확률을 저장할 배열

    # (TODO) Write your code here
    while not converged and iteration < max_iterations:
        # 이전 파라미터 저장 (변화 확인용)
        prev_mus = np.copy(mus)
        prev_covs = np.copy(covs)
        prev_pis = np.copy(pis)
        # 1. 각 데이터 포인트 x에 대해 해당 포인트가 각 cluster에 속할 확률 계산.
        for i in range (dataSize):
            xi = X[i]
            probabilities = np.zeros(k)
            for alpha in range(k):
                likelihood = multivariate_normal_pdf(xi, mus[alpha], covs[alpha])
                probabilities[alpha] = likelihood * pis[alpha]
            
            # Posterior 확률 계산하여 저장
            memberships[i] = probabilities / np.sum(probabilities)

        # 2. parameter 업데이트
        for alpha in range(k):
            Nk = np.sum(memberships[:, alpha]) # Posterior 확률의 합 (분모)
            diff = X - mus[alpha]  # 각 데이터 포인트와 평균의 차이
        
            mus[alpha] = np.sum(memberships[:, alpha].reshape(-1, 1) * X, axis=0) / Nk  # 평균 업데이트
            pis[alpha] = Nk / dataSize # 클러스터 비율 업데이트
            covs[alpha] = np.dot((memberships[:, alpha].reshape(-1, 1) * diff).T, diff) / Nk # 공분산 업데이트
            
        # 파라미터 변화량 확인
        mus_change = np.linalg.norm(mus - prev_mus)
        covs_change = np.linalg.norm(covs - prev_covs)
        pis_change = np.linalg.norm(pis - prev_pis)

        # 변화량이 tolerance 이하인 경우 수렴했다고 판단
        if mus_change < tolerance and covs_change < tolerance and pis_change < tolerance:
            converged = True

        # 반복 횟수 증가
        iteration += 1

    # 1~2를 parameter의 변화가 매우 작아지거나 최대 반복 횟수에 도달했을 때까지 반복.

    # parameter는 EM 알고리즘을 사용해서 estimate

    # 반환 파라미터
    # centroids: 클러스터 중심점
    # memberships: a list of memberships for data points(각 데이터 포인트의 클러스터 할당)
    return mus, memberships


# 다변량 정규 분포 PDF 계산
def multivariate_normal_pdf(x, mean, cov):
    d = x.shape[0]  # 데이터의 차원
    cov_det = np.linalg.det(cov)  # 공분산 행렬의 행렬식
    cov_inv = np.linalg.inv(cov)  # 공분산 행렬의 역행렬
    x_mu = x - mean
    
    coeff = 1 / ((2 * np.pi) ** (d / 2) * cov_det ** 0.5)
    exponent = -0.5 * np.dot(x_mu.T, np.dot(cov_inv, x_mu))
    
    return coeff * np.exp(exponent)



def dbscan(X, eps, min_pts):
    # 입력 파라미터 설명
    # X: 학습 데이터 포인트
    # eps: 반경
    # min_pts: 최소 데이터 포인트 수 

    # (TODO) Write your code here
    # 초기화 : 각 데이터 포인트에 대해 아직 분류되지 않은 상태로 시작
    # 각 데이터 포인트 i에 대해 다음 단계 수행
    # 1. o가 cluster에 할당되지 않은 경우
        # 1-1. 데이터가 core-object인지 확인
            # 1-1-1. 밀도 도달 가능한 객체를 모은다.
            # 1-1-2. 새로운 cluster에 할당
        # 1-2. core-object가 아니라면
            #1-2-1. 해당 포인트를 noise로 할당.
    # 위 과정을 모든 데이터 포인트가 cluster에 할당되거나 noise로 분류될 때까지 반복.

    # 반환 파라미터
    # memberships: a list of memberships for data points(각 데이터 포인트의 클러스터 할당)
    return memberships


# 메인 코드 블록
if __name__ == "__main__":
    # 선택된 데이터 유형에 따라 데이터를 생성.
    X, y = generate_data(args.data_type)

    # 선택된 클러스터링 방법에 따라 알고리즘을 실행
    if args.cluster_method == 0:
        centroids_kmeans, memberships_kmeans = kmeans(X, 100, args.cluster_num)
    elif args.cluster_method == 1:
        centroids_kmeans, memberships_kmeans = gmm(X, 100, args.cluster_num)
    elif args.cluster_method == 2:
        memberships_kmeans = dbscan(X, args.eps, args.min_pts)
    else:
        pass

    # 결과를 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("K = " + str(args.cluster_num))
    ax.scatter(X[:, 0], X[:, 1], c=memberships_kmeans, s=50, cmap='Paired')

    # 클러스터 중심점을 시각화
    if args.cluster_method == 0 or 1:
        ax.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], color='red', s=50, marker="X")
    plt.show()



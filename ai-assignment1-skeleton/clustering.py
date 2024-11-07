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
        plt.show()

        return X, y

    else:
        # 더블 문(half-moon) 모양의 데이터를 생성
        # Data type 2: double moon shaped data
        X, y = make_moons(200, noise=.05, random_state=0)
        print (X)

        # 생성된 데이터 시각화
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='Paired')
        plt.show()

        return X, y



def kmeans(X, max_iterations, k):
    # 입력 파라미터 설명
    # X: 학습 데이터 포인트
    # max_iterations: 최대 반복 횟수
    # k: 클러스터의 개수

    # (TODO) Write your code here
    

    # 반환 파라미터
    # centroids: 클러스터 중심점
    # memberships: a list of memberships for data points(각 데이터 포인트의 클러스터 할당)
    return centroids, memberships


def gmm(X, max_iterations, k):
    # 입력 파라미터 설명
    # X: 학습 데이터 포인트
    # max_iterations: 최대 반복 횟수
    # k: 클러스터의 개수

    # (TODO) Write your code here

    # 반환 파라미터
    # centroids: 클러스터 중심점
    # memberships: a list of memberships for data points(각 데이터 포인트의 클러스터 할당)
    return centroids, memberships


def dbscan(X, eps, min_pts):
    # 입력 파라미터 설명
    # X: 학습 데이터 포인트
    # eps: 반경
    # min_pts: 최소 데이터 포인트 수 

    # (TODO) Write your code here

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
        ax.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], color='red', s=50, marker="X");
    plt.show()


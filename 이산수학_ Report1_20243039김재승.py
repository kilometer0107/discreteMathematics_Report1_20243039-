# 조건 1. 형렬 입력 기능
def get_matrix_input():
    try:
        n = int(input("행렬의 크기 n을 입력하세요: "))
        if n <= 0:
            print("오류: 행렬의 크기는 1 이상이어야 합니다.")
            return None
        print(f"{n}x{n} 행렬 A의 원소를 행 단위로 입력하세요 (예: 1 2 3).")
        matrix = []
        for i in range(n):
            row = [float(x) for x in input(f"{i+1}번째 행 입력: ").split()]
            if len(row) != n:
                print(f"오류: {n}개의 원소를 입력해야 합니다.")
                return None
            matrix.append(row)
        return matrix
    except (ValueError, IndexError):
        print("오류: 입력 형식이 잘못되었습니다.")
        return None

def print_matrix(matrix, title=""):
    print(f"\n--- {title} ---")
    
    for row in matrix:
            print("  ".join(f"{val:8.3f}" for val in row))

    print("-" * (len(title) + 8))

# 조건 2. 행렬식을 이용한 역행렬 계산 기능
def get_minor(matrix, i, j):
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def determinant(matrix):
    n = len(matrix)
    if n == 1: return matrix[0][0]
    if n == 2: return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(n):
        det += ((-1)**j) * matrix[0][j] * determinant(get_minor(matrix, 0, j))
    return det

def inverse_by_determinant(matrix):
    det = determinant(matrix)
    if det == 0:
        print("행렬식이 0이므로 행렬식을 통해 역행렬 계산이 불가능합니다.")
        return None
    n = len(matrix)
    if n == 2:
        return [[matrix[1][1]/det, -matrix[0][1]/det],
                [-matrix[1][0]/det, matrix[0][0]/det]]
    cofactors = []
    for r in range(n):
        cofactor_row = []
        for c in range(n):
            minor = determinant(get_minor(matrix, r, c))
            cofactor_row.append(((-1)**(r+c)) * minor)
        cofactors.append(cofactor_row)
    adjugate = [[cofactors[j][i] for j in range(n)] for i in range(n)]
    return [[elem / det for elem in row] for row in adjugate]

# 조건 3. 가우스-조던 소거법(Gauss-Jordan elimination)을 이용한 역행렬 계산 기능
def inverse_by_gauss(matrix):
    n = len(matrix)
    identity = [[float(i == j) for i in range(n)] for j in range(n)]

    for i in range(n):
        pivot = i
        while pivot < n and matrix[pivot][i] == 0: pivot += 1
        if pivot == n: 
            print("행렬식이 0이므로 가우스-조던 소거법을 통해 역행렬 계산이 불가능합니다.")
            return None
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        identity[i], identity[pivot] = identity[pivot], identity[i]

        divisor = matrix[i][i]
        for j in range(i, n): matrix[i][j] /= divisor
        for j in range(n): identity[i][j] /= divisor

        for k in range(n):
            if i != k:
                multiplier = matrix[k][i]
                for j in range(i, n): matrix[k][j] -= multiplier * matrix[i][j]
                for j in range(n): identity[k][j] -= multiplier * identity[i][j]
    return identity


# 추가기능 : 연립방정식 입력 기능
def get_vector_input(size):
    try:
        print(f"{size}x1 벡터 b의 원소를 입력하세요 (예: 1 2 3).")
        vec_input = input("벡터 입력: ").split()
        if len(vec_input) != size:
            print(f"오류: {size}개의 원소를 입력해야 합니다.")
            return None
        vector = [float(x) for x in vec_input]
        return vector
    except (ValueError, IndexError):
        print("오류: 입력 형식이 잘못되었습니다.")
        return None

def print_input_vector(vector, title=""):
    print(f"\n--- {title} ---")
    if vector is None:
        print("벡터가 존재하지 않습니다.")
    else:
        for i, val in enumerate(vector):
            print(f"  b{i+1} = {val:8.3f}")
    print("-" * (len(title) + 8))

def print_solution_vector(vector, title=""):
    print(f"\n--- {title} ---")
    if vector is None:
        print("해를 구할 수 없습니다.")
    else:
        for i, val in enumerate(vector):
            print(f"  x{i+1} = {val:8.3f}")
    print("-" * (len(title) + 8))
    
# 추가기능 : 선형 연립방정식 풀이
def solve_linear_system(matrix, vector_b):
    vec = vector_b[:]
    n = len(matrix)

    for i in range(n):
        pivot = i
        while pivot < n and matrix[pivot][i] == 0: pivot += 1
        if pivot == n: 
            print("해가 없거나 무수히 많습니다.")
            return None
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        vec[i], vec[pivot] = vec[pivot], vec[i] 

        
        divisor = matrix[i][i]
        for j in range(i, n): matrix[i][j] /= divisor
        vec[i] /= divisor 

        
        for k in range(n):
            if i != k:
                multiplier = matrix[k][i]
                for j in range(i, n): matrix[k][j] -= multiplier * matrix[i][j]
                vec[k] -= multiplier * vec[i] 
    
    return vec 

# 조건 4. 결과 출력 및 비교
def compare(m1, m2):
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            if abs(m1[i][j] - m2[i][j]) > 1e-9: return False
    return True

if __name__ == "__main__":
    while True:
        print("\n" + "="*30)
        print("  메뉴를 선택하세요:")
        print("  1. 역행렬 계산")
        print("  2. 선형 연립방정식 풀이 (Ax=b)")
        print("  3. 종료")
        print("="*30)
        choice = input("선택: ")

        if choice == '1':
            user_matrix = get_matrix_input()
            if user_matrix:
                import copy
                matrix_for_det = copy.deepcopy(user_matrix)
                matrix_for_gauss = copy.deepcopy(user_matrix)

                
                print_matrix(user_matrix, "입력된 행렬 A")

                inv_det = inverse_by_determinant(matrix_for_det)
                if inv_det is not None:
                    print_matrix(inv_det, "행렬식을 이용한 역행렬")

                inv_g = inverse_by_gauss(matrix_for_gauss)
                if inv_g is not None:
                    print_matrix(inv_g, "가우스-조던 소거법을 이용한 역행렬")
                print("\n--- 최종 결과 비교 ---")

                if inv_det is None and inv_g is None:
                    print("두 방법 모두 역행렬이 존재하지 않음을 확인했습니다.")
                elif compare(inv_det, inv_g):
                    print("두 방법으로 계산한 역행렬이 동일합니다.")
                else:
                    print("두 방법으로 계산한 역행렬이 다릅니다.")

        elif choice == '2':
            matrix_A = get_matrix_input()
            if matrix_A:
                vector_b = get_vector_input(len(matrix_A))
                if vector_b:
                    print_matrix(matrix_A, "입력된 행렬 A")
                    print_input_vector(vector_b, "입력된 벡터 b")
                    solution_x = solve_linear_system(matrix_A, vector_b)
                    print_solution_vector(solution_x, "연립방정식의 해 x")

        elif choice == '3':
            import sys
            print("프로그램을 종료합니다.")
            sys.exit()

        else:
            print("잘못된 선택입니다. 1, 2, 3 중에서 선택하세요.")
        
        input("\n계속하려면 Enter 키를 누르세요...")
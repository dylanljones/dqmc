! Created by Dylan Jones on 30/12/2021.


subroutine matrix_product_sequence_0beta(L, N, MATS, S, SEQS, T)

implicit none

    integer :: L, N, S, K, T
    double precision, dimension(*) :: MATS(L, N, N)
    double precision, dimension(*) :: SEQS(S, N, N)
    double precision, dimension(*) :: TMP(N, N), C(N, N)
    integer, dimension(*) :: INDICES(L)
    integer :: i, j, i0, i1

    ! Number of elements in each matrix product
    K = int(L / S)
    ! Build shifted index array
    do i=1, L
        j = i + T - 1
        if (j > L) j = modulo(j, L)
        INDICES(i) = j
    end do
    ! Multiply products and store in Sequence
    do i=1, S
        i0 = ((i-1) * K) + 1
        i1 = i * K
        i0 = i1-K+1
        TMP(:, :) = MATS(INDICES(i0), :, :)
        do j = i0+1, i1
            T = INDICES(j)
            call dgemm("N", "N", N, N, N, 1.d0, MATS(T, :, :), N, TMP, N, 0.d0, C, N)
            TMP = C
        end do
        SEQS(i, :, :) = TMP(:, :)
    end do

end subroutine matrix_product_sequence_0beta


subroutine matrix_product_sequence_beta0(L, N, MATS, S, SEQS, T)

    implicit none

    integer :: L, N, S, K, T
    double precision, dimension(*) :: MATS(L, N, N)
    double precision, dimension(*) :: SEQS(S, N, N)
    double precision, dimension(*) :: TMP(N, N), C(N, N)
    integer, dimension(*) :: INDICES(L)
    integer :: i, j, i0, i1

    ! Number of elements in each matrix product
    K = int(L / S)
    ! Build shifted index array
    do i=1, L
        j = i + T - 1
        if (j > L) j = modulo(j, L)
        INDICES(i) = j
    end do
    ! Multiply products and store in Sequence
    do i=1, S
        i0 = ((i-1) * K) + 1
        i1 = i * K
        i0 = i1-K+1
        TMP(:, :) = MATS(INDICES(i0), :, :)
        do j = i0+1, i1
            T = INDICES(j)
            call dgemm("N", "N", N, N, N, 1.d0, TMP, N, MATS(T, :, :), N, 0.d0, C, N)
            TMP = C
        end do
        SEQS(S-i+1, :, :) = TMP(:, :)
    end do

end subroutine matrix_product_sequence_beta0


subroutine timeflow_map(L, N, TSM, Q, D, T, TAU, LWORK, INFO)

    implicit none

    integer :: L, N

    double precision, dimension(*) :: TSM(L, N, N)
    double precision, dimension(*) :: Q(N, N)
    double precision, dimension(*) :: D(N)
    double precision, dimension(*) :: T(N, N)
    double precision, dimension(*) :: TAU(N)
    integer, intent(inout) :: LWORK
    double precision, dimension(*) :: W(N, N), TMP(N, N)
    double precision, dimension(*) :: NORM(N)
    double precision, dimension(:), allocatable :: WORK
    integer, dimension(*) :: JPVT(N)
    integer :: INFO, i, j, k
    double precision :: dnrm2

    ! First QR decomposition with pivoting
    ! ------------------------------------

    Q(:, :) = TSM(1, :, :)
    JPVT(:) = 0  ! mark all columns as free
    ! Perform optimal LWORK query by calling DGEQP3 with LWORK=-1
    LWORK = -1
    allocate(WORK(1))
    call dgeqp3(N, N, Q, N, JPVT, TAU, WORK, LWORK, INFO)
    ! Perform QR decomposition with optimal LWORK
    LWORK = INT(WORK(1))
    deallocate(WORK)
    allocate(WORK(LWORK))
    call dgeqp3(N, N, Q, N, JPVT, TAU, WORK, LWORK, INFO)

    ! Initialize T matrix product
    ! ---------------------------

    ! Extract diagonal elements of R (upper triangular matrix of Q)
    do i=1, N
        D(i) = Q(i, i)
        if (D(i) == 0.d0) D(i) = 1.d0
    end do
    ! Construct matrix T_1 and store in T
    T(:, :) = 0.d0
    do j=1, N
        do i=1, j
            T(i, JPVT(j)) = (1 / D(i)) * Q(i, j)
        end do
    end do

    do k=2, L
        ! Store next time step matrix in temporary array W
        W(:, :) = TSM(k, :, :)

        ! Build W matrix
        ! --------------

        ! Multiply with Q from the right, overwriting the 'W' matrix
        call dormqr("R", "N", N, N, N, Q, N, TAU, W, N, WORK, LWORK, INFO)
        ! Scale by previous diagonal entries
        do i=1, N
            call dscal(N, D(i), W(:, i), 1)
        end do

        ! Compute pivot indices
        ! ---------------------

        ! Compute column norms
        do j=1, N
            NORM(j) = dnrm2(N, W(:, j), 1)
        end do
        ! Determine jpvt with an insertion sort
        JPVT(1) = 1
        do i = 2, N
            j = i - 1
            do while (j >= 1)
                if (NORM(JPVT(j)) >= NORM(i)) exit
                JPVT(j + 1) = JPVT(j)
                j = j - 1
            end do
            JPVT(j + 1) = i
        end do

        ! Pre-pivot W
        !------------

        ! store pre-pivoted W in Q
        do j=1, N
            Q(:, j) = W(:, JPVT(j))
        end do
        ! Perform QR decomposition on Q
        call dgeqrf(N, N, Q, N, TAU, WORK, LWORK, INFO)

        ! Accumulate T matrix product
        ! ---------------------------

        ! Extract diagonal elements of R (upper triangular matrix of Q)
        do i=1, N
            D(i) = Q(i, i)
            if (D(i) == 0.d0) D(i) = 1.d0
        end do
        ! Multiply inverse diagonal matrix with the upper triangular R matrix
        W(:, :) = 0.d0
        do j=1, N
            do i=1, j
                W(i, j) = (1 / D(i)) * Q(i, j)
            end do
        end do
        ! Apply pivoting permutation to the rows of previous T
        do i=1, N
            TMP(i, :) = T(JPVT(i), :)
        end do
        T(:, :) = TMP(:, :)
        ! multiply with upper triangular matrix, overwriting 'T' in-place;
        call dtrmm("L", "U", "N", "N", N, N, 1.0d0, W, N, T, N)

    end do

end subroutine timeflow_map


subroutine timeflow_map_0beta(L, N, TSM, S, TIDX, Q, D, T, TAU, LWORK, INFO)

    implicit none

    integer :: L, N, S, TIDX
    double precision, dimension(*) :: TSM(L, N, N)
    double precision, dimension(*) :: Q(N, N)
    double precision, dimension(*) :: D(N)
    double precision, dimension(*) :: T(N, N)
    double precision, dimension(*) :: TAU(N)
    integer, intent(inout) :: LWORK, INFO

    double precision, dimension(*) :: SEQS(S, N, N)
    double precision, dimension(*) :: TMP(N, N), C(N, N)
    integer, dimension(*) :: INDICES(L)
    integer :: i, j, i0, i1, K

    ! Pre compute matrix product sequence
    ! -----------------------------------

    ! Number of elements in each matrix product
    K = int(L / S)
    ! Build shifted index array
    do i=1, L
        j = i + TIDX - 1
        if (j > L) j = modulo(j, L)
        INDICES(i) = j
    end do
    ! Multiply products and store in Sequence
    do i=1, S
        i0 = ((i-1) * K) + 1
        i1 = i * K
        i0 = i1-K+1
        TMP(:, :) = TSM(INDICES(i0), :, :)
        do j = i0+1, i1
            TIDX = INDICES(j)
            call dgemm("N", "N", N, N, N, 1.d0, TSM(TIDX, :, :), N, TMP, N, 0.d0, C, N)
            TMP = C
        end do
        SEQS(i, :, :) = TMP(:, :)
    end do

    ! Compute timeflow map
    ! --------------------
    call timeflow_map(S, N, SEQS, Q, D, T, TAU, LWORK, INFO)

end subroutine timeflow_map_0beta


subroutine timeflow_map_beta0(L, N, TSM, S, TIDX, Q, D, T, TAU, LWORK, INFO)

    implicit none

    integer :: L, N, S, TIDX
    double precision, dimension(*) :: TSM(L, N, N)
    double precision, dimension(*) :: Q(N, N)
    double precision, dimension(*) :: D(N)
    double precision, dimension(*) :: T(N, N)
    double precision, dimension(*) :: TAU(N)
    integer, intent(inout) :: LWORK, INFO

    double precision, dimension(*) :: SEQS(S, N, N)
    double precision, dimension(*) :: TMP(N, N), C(N, N)
    integer, dimension(*) :: INDICES(L)
    integer :: i, j, i0, i1, K

    ! Pre compute matrix product sequence
    ! -----------------------------------

    ! Number of elements in each matrix product
    K = int(L / S)
    ! Build shifted index array
    do i=1, L
        j = i + TIDX - 1
        if (j > L) j = modulo(j, L)
        INDICES(i) = j
    end do
    ! Multiply products and store in Sequence
    do i=1, S
        i0 = ((i-1) * K) + 1
        i1 = i * K
        i0 = i1-K+1
        TMP(:, :) = TSM(INDICES(i0), :, :)
        do j = i0+1, i1
            TIDX = INDICES(j)
            call dgemm("N", "N", N, N, N, 1.d0, TMP, N, TSM(TIDX, :, :), N, 0.d0, C, N)
            TMP = C
        end do
        SEQS(S-i+1, :, :) = TMP(:, :)
    end do

    ! Compute timeflow map
    ! --------------------
    call timeflow_map(S, N, SEQS, Q, D, T, TAU, LWORK, INFO)

end subroutine timeflow_map_beta0

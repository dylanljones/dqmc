! Created by Dylan Jones on 30/12/2021.


subroutine construct_greens(N, Q, D, T, G, TAU, SIGN, LOGDET, LWORK, INFO)

    implicit none
    integer :: N
    double precision, dimension(*) :: Q(N, N)
    double precision, dimension(*) :: D(N)
    double precision, dimension(*) :: T(N, N)
    double precision, dimension(*) :: G(N, N)
    double precision, dimension(*) :: TAU(N)
    integer, intent(inout) :: SIGN
    double precision, intent(inout) :: LOGDET
    integer, intent(inout) :: LWORK
    integer, intent(inout) :: INFO
    double precision, dimension(:) :: WORK(LWORK)
    double precision, dimension(*) :: TMP(N)
    integer, dimension(*) :: JPVT(N)
    integer :: i
    double precision :: tel

    ! Construct the matrix D_b^{-1}
    do i=1, N
        if (abs(D(i)) > 1.d0) then
            G(i, i) = 1.d0 / D(i)
        else
            G(i, i) = 1.d0
        end if
    end do

    ! calculate D_b^{-1} Q^T
    call dormqr("R", "T", N, N, N, Q, N, TAU, G, N, WORK, LWORK, INFO)
    ! Calculate D_s T and store result in T
    do i=1, N
        TMP(:) = T(i, :)
        if (abs(D(i)) <= 1.d0) call dscal(N, D(i), TMP, 1)
        T(i, :) = TMP(:)
    end do
    ! Calculate D_b^{-1} Q^T + D_s T, store result in T
    call daxpy(N*N, 1.d0, G, 1, T, 1)
    ! Perform a LU decomposition of D_b^{-1} Q^T + D_s T
    call dgetrf(N, N, T, N, JPVT, INFO)

    ! Calculate sign/determinant of (D_b^{-1} Q^T + D_s T)^{-1} (D_b^{-1} Q^T)
    SIGN = 1
    LOGDET = 0.d0
    ! Contribution from (D_b^{-1} Q^T + D_s T)^{-1}
    do i=1, N
        tel = T(i, i)
        if (tel < 0.d0) SIGN = -SIGN
        LOGDET = LOGDET - log(abs(tel))
        if (JPVT(i) .ne. i) SIGN = -SIGN
    end do
    ! Contribution from D_b^{-1}
    do i=1, N
        if (abs(D(i)) > 1.d0) then
            tel = D(i)
            if (tel < 0) SIGN = -SIGN
            LOGDET = LOGDET - log(abs(tel))
        end if
    end do
    ! Q contributes a factor 1 or -1 to determinant
    do i=1, N
        if (TAU(I) > 0.d0) SIGN = -SIGN
    end do
    ! Calculate (D_b^{-1} Q^T + D_s T)^{-1} (D_b^{-1} Q^T) and overwrite gf
    ! DGETRS solves a system of linear equations A * X = B with a general N-by-N
    ! matrix A using the LU factorization computed by DGETRF.
    call dgetrs("N", N, N, T, N, JPVT, G, N, INFO)

end subroutine construct_greens


subroutine update_greens(L, N, NU, CONFIG, G_UP, G_DN, i, t)

    implicit none

    integer, intent(in) :: N, L
    double precision, intent(in) :: NU
    integer, dimension(*), intent(in) :: CONFIG(N, L)
    double precision, dimension(*), intent(inout) :: G_UP(N, N), G_DN(N, N)
    integer, intent(in) :: i, t
    double precision :: tmp, alpha, ONE
    double precision, dimension(*) :: U(N), W(N)

    ONE = 1.d0
    tmp = exp(-2.d0 * NU * CONFIG(i, t))

    ! Spin-up
    alpha = tmp - ONE
    ! Copy i-th column of GF and form (G-I) e_i
    U(:) = G_UP(:, i)
    U(i) = U(i) - ONE
    ! Copy i-th row of GF
    W(:) = G_UP(i, :)
    ! Compute G = a U W^T + G
    call dger(N, N, alpha / (ONE - alpha * U(i)), U, 1, W, 1, G_UP, N)

    ! Spin-down
    alpha = (1 / tmp) - ONE
    ! Copy i-th column of GF and form (G-I) e_i
    U(:) = G_DN(:, i)
    U(i) = U(i) - ONE
    ! Copy i-th row of GF
    W(:) = G_DN(i, :)
    ! Compute G = a U W^T + G
    call dger(N, N, alpha / (ONE - alpha * U(i)), U, 1, W, 1, G_DN, N)

end subroutine update_greens



subroutine wrap_up_greens(L, N, BMATS, BMATS_INV, G, t)

    implicit none

    integer, intent(in) :: N, L
    double precision, dimension(*), intent(in) :: BMATS(L, N, N), BMATS_INV(L, N, N)
    double precision, dimension(*), intent(inout) :: G(N, N)
    integer, intent(in) :: t
    double precision, dimension(*) :: TMP(N, N)
    double precision :: alpha, beta

    alpha = 1.d0
    beta = 0.d0
    ! Compute B * G
    call dgemm("N", "N", N, N, N, alpha, BMATS(t, :, :), N, G, N, beta, TMP, N)
    ! Compute (B * G) * B^{-1}
    call dgemm("N", "N", N, N, N, alpha, TMP, N, BMATS_INV(t, :, :), N, beta, G, N)

end subroutine wrap_up_greens



subroutine wrap_down_greens(L, N, BMATS, BMATS_INV, G, t)

    implicit none

    integer, intent(in) :: N, L
    double precision, dimension(*), intent(in) :: BMATS(L, N, N), BMATS_INV(L, N, N)
    double precision, dimension(*), intent(inout) :: G(N, N)
    integer, intent(in) :: t
    double precision, dimension(*) :: TMP(N, N)
    double precision :: alpha, beta

    alpha = 1.d0
    beta = 0.d0
    ! Compute B^{-1} * G
    call dgemm("N", "N", N, N, N, alpha, BMATS_INV(t-1, :, :), N, G, N, beta, TMP, N)
    ! Compute (B^{-1} * G) * B
    call dgemm("N", "N", N, N, N, alpha, TMP, N, BMATS(t-1, :, :), N, beta, G, N)

end subroutine wrap_down_greens

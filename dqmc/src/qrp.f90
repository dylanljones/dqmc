! Created by  on 28.12.21.

subroutine qp3(M, N, A, JPVT)

    implicit none

    integer :: M, N

    double precision, dimension(*) :: A(M, N)
    double precision, dimension(*) :: TAU(M)

    double precision, dimension(:), allocatable :: WORK
    integer, dimension(*) :: JPVT(N)
    integer :: LDA, LWORK, INFO

    ! Setup parameters
    LDA = M
    JPVT(:) = 0  ! mark all columns as free

    ! Perform optimal LWORK query by calling DGEQP3 with LWORK=-1
    LWORK = -1
    allocate(WORK(1))
    call dgeqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)

    ! Perform QR decomposition with optimal LWORK
    LWORK = INT(WORK(1))
    deallocate(WORK)
    allocate(WORK(LWORK))
    call dgeqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)

end subroutine qp3




subroutine qrp(M, N, A, R, JPVT)

    implicit none

    integer :: M, N

    double precision, dimension(*) :: A(M, N), R(M, N)
    double precision, dimension(*) :: TAU(M)

    double precision, dimension(:), allocatable :: WORK
    integer, dimension(*) :: JPVT(N)
    integer :: LDA, LWORK, INFO, i, j


    ! Setup parameters
    LDA = M
    JPVT(:) = 0  ! mark all columns as free

    ! Perform optimal LWORK query by calling DGEQP3 with LWORK=-1
    LWORK = -1
    allocate(WORK(1))
    call dgeqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)

    ! Perform QR decomposition with optimal LWORK
    LWORK = INT(WORK(1))
    deallocate(WORK)
    allocate(WORK(LWORK))
    call dgeqp3(M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO)

    R(:, :) = 0.d0
    ! Construct Upper triangular matrix R from QR
    do i=1, M
        do j=i, N
            R(i, j) = A(i, j)
        end do
    end do

    ! Compute matrix Q and store in A
    call dorgqr(M, N, N, A, LDA, TAU, WORK, LWORK, INFO)

end subroutine qrp
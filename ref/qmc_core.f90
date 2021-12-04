!========+=========+=========+=========+=========+=========+=========+=$
! PROGRAM: lisaqmc.f
! NOTICE : This program accompanies the revue article:
!
!           The Local Impurity Self Consistent Approximation (LISA)
!                   to Strongly Correlated Fermion Systems
!                   and the Limit of Infinite Dimensions
!
!                                   by
!
!          A. Georges, G. Kotliar, W. Krauth, M. Rozenberg
!
!            to be published in: Reviews of Modern Physics (1996)
!
!          (the paper will be referred to as ``GKKR'').
!          you are kindly asked to cite the paper (and, if applicable,
!              the original works) if you use this program.
!
!          The programs have  been thoroughly tested on SUN Sparks,
!          HP 900, IBM RS6000 stations.
!
! TYPE   : main
! PURPOSE: program for qmc-simulation of Anderson impurity
!          problem
! I/O    : cf file README_lisaqmc
! VERSION: 28-07-03  Xmu added
! AUTHOR : W. Krauth (krauth@physique.ens.fr)
! COMMENT: Even though FORTRAN is case-insensitive, I have capitalized
!          all the global variables, i. e. the variables appearing
!          in the common/global/ block (cf file lisaqmc.dat).
!          At the end of the program, a number of SLATEC routines
!          have been appended. You may not want to  print these.
!========+=========+=========+=========+=========+=========+=========+=$
      subroutine qmc(maxt,nwarm)
      include 'param.dat'
      real ranw
      logical stochastic
      dimension greenm(-L+1:L-1),greenm2(-L+1:L-1),greenupsum(0:L-1)
      dimension greendosum(0:L-1)
      dimension chipm(-L+1:L-1)
      integer itau(0:L+1),iss(L)
      nran(i)=mod(int(i*ranw(Idum)),i) + 1
!     Zero=0
!     One=1
!     Two=2
      stochastic=.true.
      if (L.le.30) stochastic=.false.
!     if (L.le.16) stochastic=.false.
!     open (unit=1,file='lisaqmc.input',form='formatted',status='old')
!c
!c    write header onto standard output
!c
!     call wheader(6)
!======================================================================
!     initial set-up
!======================================================================
!     read (1,*)Beta,U,xmut,H
!c
!c    cosh(Xlambda)=exp(deltau*U/2)     cf. eq. (\ref{hirschdecoup})
!c
      deltau=Beta/real(L)
      xmud=deltau*xmu
      dummy=exp(deltau*Uo/Two)
      Xlambda=log(dummy+sqrt(dummy**2-One))
      print*,'U, Lambda =',Uo,Xlambda
      print*,'L, Beta, deltau =',L,Beta,deltau
      Idum=-75949
      Idum=int(-ranw(Idum)*100000)
!     read (1,*)maxt
!     read (1,*)ic,jc
      if(maxt.eq.0) maxt=100000
      ic=0
      jc=1
      if (jc.eq.0.and.abs(Hmag).lt.1.e-7) then
         Paramagnet=.true.
      else
         Paramagnet=.false.
      end if
!     open (unit=14,file='lisaqmc.init',form='formatted',status='old')
!     print*,' begin initi paa', paramagnet
!--------  Change Green0t(L) to Green(L,L') !
      call initial(xmud)
!     print*,' end init'
!      do i=1,L
!            write(*,'(8f10.5)')(green0up(i,j),j=1,8)
!      end do
!      print*, ' green0do'
!      do i=1,L
!            write(*,'(8f10.5)')(green0do(i,j),j=1,8)
!      end do
      naccept=0
      isign=1
      do i=0,L-1
         chipm(i)=Zero
      enddo

      do 810 i=0,L-1
         greenupsum(i)=Zero
         greendosum(i)=Zero
810   continue
      dsum=Zero
      upsum=Zero
      dosum=Zero
      iitime=0
      iter=0
!***********************************************************************
!		start simulation
!***********************************************************************
!c
!c    SIMULATION BY MONTE CARLO   (stochastic.eq.true)
!c
      if (stochastic) then
         nfastup=100
         do 1000 iii=1,1000000000
!c
!c       we make nfastup sweeps with the fast update (eq. (\ref{fastupdate}))
!c       until checking with subroutine 'update' (eq.
!c       whether precision hasn't
!c       deteriorated. If that is the case, we make nfastup smaller,
!c       otherwise bigger.
!c
         do 2000 kkk=1,nfastup
         if (iter.ge.maxt) goto 3120
         iter=iter+1
         do 1900 kk=1,L
            k=nran(L)
!
!           try flipping spin k
!
            isignnew=abs(detrat(k))/detrat(k)
            if (isignnew.ne.isign) then
               print*,'signchange',isignnew
               isign=isignnew
            end if
            dummy=abs(detrat(k))
            if (ranw(Idum).lt.dummy/(One+dummy)) then
!c
!c             accept flip,
!c
               call record(k)
               naccept=naccept+1
            end if
1900        continue
!c
!c          end of sweep: calculation of green's function
!c
!------------  "warm-up" Ising spins!!!!!
            if(iter.lt.nwarm) go to 2000
            iitime=iitime+1
            do 192 idel=-L+1,L-1
               greenm(idel)=Zero
               greenm2(idel)=Zero
               inumb=min(L-idel,L)-max(1,1-idel)+1
               do 193 i=max(1,1-idel),min(L-idel,L)
                  dummy=inumb
                  greenm(idel)=greenm(idel)+greenup(i+idel,i)/dummy
                  greenm2(idel)=greenm2(idel)+greendo(i+idel,i)/dummy
!------------ Calculate Chipm=<S+(t)S-(0)>
!         chipm(idel)=chipm(idel)-(greenup(idel+i,i)*greendo(i,idel+i)+  ! susceptibility
!     &   greendo(idel+i,i)*greenup(i,idel+i))/Two/dummy
193            continue
192         continue
            do 191 i=1,L
               dummy=L
               dsum=dsum+greenup(i,i)*greendo(i,i)/dummy
191         continue
            dosum=dosum+greenm2(0)
            upsum=upsum+greenm(0)
            do 818 i=0,L-1
               greenupsum(i)=greenupsum(i)+greenm(i)
               greendosum(i)=greendosum(i)+greenm2(i)
818         continue
2000     continue
!c
!c       update greens functions from scratch
!c
         diff=-100000
         call update(xmud)
         do 887 i=1,L
            do 887 j=1,L
               diff=max(diff,abs(Greenup(i,j)-Gnewup(i,j)),abs(Greendo(i,j)-Gnewdo(i,j)))
               Greenup(i,j)=Gnewup(i,j)
               Greendo(i,j)=Gnewdo(i,j)
887      continue
         if (diff.gt.0.0005) then
            print*, nfastup,diff,'   nfastup diff'
            nfastup=max(1,nfastup/2)
         end if
         if (diff.lt.0.0005) nfastup=nfastup*2
1000     continue
3120  continue
      else
!c
!c       exact enumeration for L <= 18  (stochastic=.false.)
!c       initialization of Gray code variables (itau is a variable used
!c       for the construction of the code cf. (Reingold  et al 1977)
!c
         do 3001 j=0,L
            itau(j)=j+1
3001     continue
         do 3002 j=1,L
            is(j)=-1
3002     continue
         call update(xmud)

         do 3199 n=1,L
            do 3199 m=1,L
               Greenup(n,m)=Gnewup(n,m)
               Greendo(n,m)=Gnewdo(n,m)
3199     continue
!        print*,' Greenup'
!        do i=1,L
!           write(*,'(8f10.5)')(Greenup(i,j),j=1,8)
!        end do
!        print*,' Greendo'
!        do i=1,L
!           write(*,'(8f10.5)')(Greendo(i,j),j=1,8)
!        end do
         do 910 i=0,L-1
            greenupsum(i)=greenup(i+1,1)
   	    greendosum(i)=greendo(i+1,1)
!------------- Start susceptibility Xi-pm
!          chipm(i)=-(greenup(i+1,1)*greendo(1,i+1)+   ! susceptibility
!     &   greendo(i+1,1)*greenup(1,i+1))/Two
910      continue
         dsum=greenup(1,1)*greendo(1,1)
         upsum=greenup(1,1)
         dosum=greendo(1,1)
         det=One
!c
!c       Note that det=One gives the determinant of the initial configuration
!c       only up to a multiplicative factor. If you are interested in the
!c       numerical value of the partition function, or the free energy,
!c       you will have to replace the above line by  the following:
!c
!c       det=One/(determinant(greenup,L)*determinant(greendo,L))
!c
!c       (the external function determinant is provided below, but not
!c       actually used
!c
         partition=det
         itup=0
      if (stochastic) then
         nexact=1
      else
         nexact=2**L-1
!        nexact=1
         print*,'exact enumeration for  N-Fileds= ',nexact+1
      endif
         do 3000 iii=1,nexact
            itup=itup+1
!c
!c          use Gray code to calculate index of spin which has to be flipped
!c
            k=itau(0)
            itau(k-1)=itau(k)
            itau(k)=k+1
            if (k.ne.1) itau(0)=1
!c
!c          flip spin k
!c
            det=detrat(k)*det
	    partition=partition+det
            isignnew=abs(detrat(k))/detrat(k)
            if (isignnew.ne.isign) then
               print*,'signchange',isignnew
               isign=isignnew
            end if
            call record(k)
            do 920 i=0,L-1
	       greenupsum(i)=greenupsum(i)+greenup(i+1,1)*det
	       greendosum(i)=greendosum(i)+greendo(i+1,1)*det
!------------- Susceptibility Xi-pm
!          chipm(i)=chipm(i)-(greenup(i+1,1)*greendo(1,i+1)+ ! susceptibility
!     &   greendo(i+1,1)*greenup(1,i+1))/Two*det
920         continue
            dsum=dsum + greenup(1,1)*greendo(1,1)*det
            upsum=upsum + greenup(1,1)*det
            dosum=dosum + greendo(1,1)*det
!c
!c          check that precision of determinant is not degraded
!c          (the last configuration of M.C. spins is again recreated from
!c          (-1,-1,...-1) and the determinant is recomputed).
!c
            if (itup.gt.100) then
               itup=0
               detcheck=One
               do 3007 j=1,L
                  iss(j)=is(j)
                  is(j)=-1
3007           continue
               call update(xmud)
               do 3799 n=1,L
                  do 3799 m=1,L
                     Greenup(n,m)=Gnewup(n,m)
                     Greendo(n,m)=Gnewdo(n,m)
3799           continue
               do 881 k=1,L
                  if (iss(k).ne.-1) then
                     detcheck=detcheck*detrat(k)
                     call record(k)
                  end if
881            continue
               det=detcheck
            end if
3000     continue
      end if
!====================================================================
!                      END OF SIMULATION
!====================================================================
      if (stochastic) then
         facnorm=iitime
      else
         facnorm=partition
      end if
!     open (unit=17,file='lisaqmc.result',form='formatted',status='new')
!     call wheader(17)
!     if (Paramagnet) then
!        do 717 i=0,L-1
!        write(17,'(f20.10)')(greenupsum(i)+greendosum(i))/facnorm/Two
717      continue
!     else
!        do 718 i=0,L-1
!           write(17,'(2f20.10)')greenupsum(i)/facnorm,
!    &      greendosum(i)/facnorm
718   continue
!     end if
!c----------Susceptibility
!        rewind(3)
!         do  i=0,L-1
!            chipm(i)=chipm(i)/facnorm
!         if(i.eq.0) chipm(i)=chipm(i)+(upsum+dosum)/facnorm/Two
!         write(3,'(3f15.8)')Beta/Real(L)*i,chipm(i)
!         enddo
!c     sign change, from QMC-definition!
         do  i=0,L-1
            Greentup(i+1)=-greenupsum(i)/facnorm
            Greentdo(i+1)=-greendosum(i)/facnorm
!            Chit(i+1)=chipm(i)  ! susceptibility
!     write(2,'(i5,2f15.8)')i,Greentup(i+1),Greentdo(i+1)
         enddo
      write(*,'(a20,f15.7)')' prob double occ=',dsum/facnorm
      write(*,'(a20,f15.7)')' density    up  =', One-upsum/facnorm
      write(*,'(a20,f15.7)')' density    do  =', One-dosum/facnorm
      if (stochastic) then
      write(6,'(a60)')'========================================'
      write(6,'(a20)')'         Summary'
      write(6,'(a20,I15)')'Length of Simul = ',iter
      write(6,'(a20,f15.4)')'Acceptance Prob = ', naccept/real(maxt*L)
      write(6,'(a60)')'========================================'
       end if
!     open (unit=15,file='lisaqmc.end',form='formatted',status='new')
!     call wheader(15)
!     if (Paramagnet) then
!        do 616 i=1,L
!           write(15,'(f20.10)')Green0up(i,1),Green0do(i,1)
616      continue
!     else
!        do 618 i=1,L
!           write(15,'(f20.10)')Green0up(i,1)
618      continue
!     end if
!     do 617 i=1,L
!        write(15,'(i5)')Is(i)
617   continue
!c
!c    write seed for next run onto file 15
!c
!     write(15,'(I10)')int(-ranw(Idum)*100000)
      end


!========+=========+=========+=========+=========+=========+=========+=$
!     PROGRAM: detrat.f
!     TYPE   : function
!     PURPOSE: calculate the ratio of the new and
!              old determinants (cf. eq. (\ref{detrat})
!     I/O    :
!     VERSION: 30-Sep-95
!     COMMENT:
!========+=========+=========+=========+=========+=========+=========+=$
      function detrat(k)
      include 'param.dat'
      rup=One+(One-Greenup(k,k))*(exp(-Two*Xlambda*real(Is(k)))-One)
      rdo=One+(One-Greendo(k,k))*(exp( Two*Xlambda*real(Is(k)))-One)
      detrat=rup*rdo
      end



!========+=========+=========+=========+=========+=========+=========+=$
!     PROGRAM: initial.f
!     TYPE   : subroutine
!     PURPOSE: read in initial configuration of bath Green's function
!              and of Ising spins, expand G(i-j) into matrix G(i,j).
!              invoke subroutine Update to calculate
!              Green's function for the initial choice of
!              Ising spins.
!     I/O    :
!     VERSION: 28-07-03  Xmu added
!     COMMENT:
!========+=========+=========+=========+=========+=========+=========+=$
      subroutine initial(xmud)
      include 'param.dat'
      dimension gtempup(-L+1:L-1)
      dimension gtempdo(-L+1:L-1)
!     print*,L,paramagnet,'= L paramagnet'
!c
!c    read in G(sigma=0) from file lisaqmc.init
!c
!     call rheader(14)
!     if (Paramagnet) then
!        do 177 i=0,L-1
!           read(14,*) gtempup(i)
!           gtempdo(i)=gtempup(i)
177      continue
!     else
!        do 171 i=0,L-1
!           read(14,*) gtempup(i),gtempdo(i)
171      continue
!     end if
!     print*,' 171'
!c     sign change, in order to conform to QMC-definition
         do 172 i=0,L-1
          gtempup(i)=-Green0tup(i+1)
          gtempdo(i)=-Green0tdo(i+1)
172      continue
!c
!c    reflection of G to calculate Greens function for neg. arguments
!c
      do 123 i=1,L-1
         gtempup(-i)=-gtempup(L-i)
         gtempdo(-i)=-gtempdo(L-i)
123   continue

!     print*,' 123'
!c
!c    choice of spins
!c
!     do 11 n=1, L
!        read(14,*) Is(n)
11    continue
!     print*,' Idum, Is'
!     write(6,'(i10)')Idum
!     write(6,'(20i3)')Is
!     read(14,*)Idum
!c
!c    calculation of Green's function
!c    update puts results in array Greenu, which is then
!c    transcribed into array Green (use translation invariance).
!c
      dummy=L
      deltau=Beta/dummy
      do 99 i=1,L
         do 99 j=1,L
            Green0up(i,j)=gtempup(i-j)
            Green0do(i,j)=gtempdo(i-j)
99    continue
      do 2 n=1,L
         do 2 m=1,L
            Greenup(n,m)=Green0up(n,m)
            Greendo(n,m)=Green0do(n,m)
2     continue
      call update(xmud)
      do 3 n=1,L
         do 3 m=1,L
            Greenup(n,m)=Gnewup(n,m)
            Greendo(n,m)=Gnewdo(n,m)
3     continue
      end


!========+=========+=========+=========+=========+=========+=========+=$
!     PROGRAM: ranw.f
!     TYPE   : real function
!     PURPOSE: produce uniformly distributed random numbers
!              following the algorithm of Mitchell and Moore
!     I/O    :
!     VERSION: 30-Sep-95
!     COMMENT: cf. D. E. Knuth, Seminumerical Algorithms, 2nd edition
!              Vol 2 of  The Art of Computer Programming (Addison-Wesley,
!              1981) pp 26f. (Note: the procedure ran3 in
!              W. H. Press et al,  Numerical
!              Recipes in FORTRAN, 2nd edition (Cambridge University
!              Press 1992)  is based on the same algorithm).
!              I would suggest that you make sure for yourself that
!              the quality of the random number generator is sufficient,
!              or else replace it!
!========+=========+=========+=========+=========+=========+=========+=$
      real function ranw(idum)
      Parameter (Mbig=2**30-2, Xinvers=1./Mbig)
      data ibit/ 1/
      Integer IX(55)
      save
      if (ibit.ne.0) then
         ibit=0
!c
!c       fill up the vector ix with some random integers, which are
!c       not all even
!c
         if (idum.eq.0) stop 'use nonzero value of idum'
         idum=abs(mod(idum,Mbig))
         ibit=0
         Ix(1)=871871
         Do i=2,55
            Ix(i)=mod(Ix(i-1)+idum,Ix(i-1))
            Ix(i)=max(mod(Ix(i),Mbig),idum)
         end do
         j=24
         k=55
!c
!c       warm up the generator
!c
         do i=1,1258
            Ix(k)=mod(Ix(k)+Ix(j),Mbig)
            j=j-1
            if (j.eq.0) j=55
            k=k-1
            if (k.eq.0) k=55
         end do
      end if
!c
!c    this is where execution usually starts:
!c
      Ix(k)=mod(Ix(k)+Ix(j),Mbig)
      j=j-1
      if (j.eq.0) j=55
      k=k-1
      if (k.eq.0) k=55
      ranw=Ix(k)*Xinvers
      end



!========+=========+=========+=========+=========+=========+=========+=$
!       PROGRAM: record
!       TYPE   : subroutine
!       PURPOSE: record changes of accepted move on the Green's function
!                (cf  eq. (\ref{fastupdate}))
!       I/O    :
!       VERSION: 30-9-95
!       COMMENT: k is the index of the spin which has been flipped
!========+=========+=========+=========+=========+=========+=========+=$
        subroutine record(k)
        include 'param.dat'
!c      update Green's function (implementation of  eq. (\ref{fastupdate}))
        del=-Two*Xlambda*real(Is(k))
        do 1 i=1,L
           idel=0
           if (i.eq.k) idel=1
           do 1 j=1,L
              Gnewup(i,j)=Greenup(i,j)+(Greenup(i,k)-idel)* (exp(del)-1.)/(1.+(1.-Greenup(k,k))*(exp(del)-One))*Greenup(k,j)
              Gnewdo(i,j)=Greendo(i,j)+(Greendo(i,k)-idel)*(exp(-del)-One)/(One+(One-Greendo(k,k))*(exp(-del)-One))*Greendo(k,j)
1       continue
        do 2 j=1,L
           do 2 i=1,L
              Greenup(i,j)=Gnewup(i,j)
              Greendo(i,j)=Gnewdo(i,j)
2       continue

!c      update spin
        Is(k)=-Is(k)
        end


!========+=========+=========+=========+=========+=========+=========+=$
!     PROGRAM: update.f
!     TYPE   : subroutine
!     PURPOSE: calculate the Green's function
!              for a given configuration of spins
!              (in vector Is) from the Green's function
!              for spins set equal to zero  (eq. (\ref{inversion}))
!     I/O    :
!     VERSION: 30-Sep-95
!     COMMENT: can be used to initialize run
!                     (subroutine initial),
!              or to check for deterioration of precision
!========+=========+=========+=========+=========+=========+=========+=$
      subroutine update(xmud)
      include 'param.dat'
      dimension a(L,L),b(L,L),ainv(L,L),binv(L,L)
!c
!c    calculate the matrix a=1-(g-1)(exp(v')-1)
!c
      do 2 i=1,L
         do 3 j=1,L
            a(i,j)=-Green0up(i,j)*(exp( Xlambda*real(Is(j))+xmud)-One)
            b(i,j)=-Green0do(i,j)*(exp(-Xlambda*real(Is(j))+xmud)-One)
3        continue
      a(i,i)=1-(Green0up(i,i)-One)*(exp( Xlambda*real(Is(i))+xmud)-One)
      b(i,i)=1-(Green0do(i,i)-One)*(exp(-Xlambda*real(Is(i))+xmud)-One)
2     continue
      call inverse(a,ainv)
      call inverse(b,binv)
      do 4 i=1,L
         do 4 j=1,L
            suma=Zero
            sumb=Zero
            do 6 k=1,L
               suma=suma+ainv(i,k)*Green0up(k,j)
               sumb=sumb+binv(i,k)*Green0do(k,j)
6           continue
         Gnewup(i,j)=suma
         Gnewdo(i,j)=sumb
4     continue
      end

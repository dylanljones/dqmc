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
!c
!c    update Green's function (implementation of  eq. (\ref{fastupdate}))
!c
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
!c
!c   update spin
!c
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
!======================================================================
! PROGRAM: inverse.f
! TYPE   : subroutine
! PURPOSE: calculate inverse of matrix
! I/O    :
! VERSION: 30-Sep-95
! COMMENT: Here we use the (public domain) SLATEC routines dgeco and
!          dgedi (which perform the Gaussian elimination) + dependencies.
!          These routines can be obtained, e.g., from netlib
!          (to learn about netlib, send an otherwise empty
!          message to netlib@research.att.com
!          containing 'send index' in the subject header, 
!          on WWW, look under the address 
!          http://netlib.att.com/netlib/master/readme.html).
!========+=========+=========+=========+=========+=========+=========+=$
      Subroutine inverse(a,y)
      include 'param.dat'
      dimension a(L,L),y(L,L)
      dimension z(L),ipvt(L)
      do 1 i=1,L
         do 2 j=1,L
              y(i,j)=a(i,j)
2        continue
1     continue
      call dgeco(y,L,L,ipvt,rcond,z)
!c
!c    we only want the Inverse, so set job = 01
!c
      job=01
      call dgedi(y,L,L,ipvt,det,z,job)
      end
!======================================================================
!       PROGRAM: rheader.f  
!       TYPE   : subroutine
!       PURPOSE: read the program's header from unit k 
!       I/O    :
!       VERSION: 30-Sep-95
!       COMMENT: 
!========+=========+=========+=========+=========+=========+=========+=$
        subroutine rheader(k) 
        do 1 i=1,10
        read(k,*)
1       continue
        end 
!========+=========+=========+=========+=========+=========+=========+=$
!       PROGRAM: wheader.f  
!       TYPE   : subroutine
!       PURPOSE: write the program's header onto standard output 
!       I/O    : 
!       VERSION: 30-Sep-95
!       COMMENT: 
!========+=========+=========+=========+=========+=========+=========+=$
        subroutine wheader(k) 
        include 'param.dat'
        character *80 xyz
        write(k,'(a55)')'========================================'
        write(k,'(a55)')' Lisaqmc  simulation: rev. 06/30/95'
        write(k,'(a55)')'========================================'
        rewind(1)
        write(k,'(4(a6,I4))') 'L=',L
        read(1,*)
        read(1,*)
        read(1,*)
        do 3 i=1,6
        read(1,'(a60)')xyz
        write(k,'(a60)')xyz
3       continue 
        rewind(1)
        end
!========+=========+=========+=========+=========+=========+=========+=$
!       PROGRAM: dasum.f daxpy.f  ddot.f dgeco.f dgedi.f dgefa.f 
!                dscal.f dswap.f idamax.f
!       TYPE   : collection of subroutines 
!       PURPOSE: calculate inverse and determinant (look at 
!                subroutine dgedi.f) 
!       I/O    :
!       VERSION: 30-Sep-95
!       COMMENT: the following subroutines are a bunch of
!                functions obtained from the slatec library
!                at Netlib, which allow the calculation of 
!                inverse and determinant.
!                You can replace these programs by the 
!                corresponding routines of your favorite library,
!                e.g. Numerical Recipes (which is not in the
!                public domain).
!                Notice that we are using the double precision 
!                versions of the programs.
!noprint=+=========+=========+=========+=========+=========+=========+=$
*DECK DASUM
      DOUBLE PRECISION FUNCTION DASUM (N, DX, INCX)
!***BEGIN PROLOGUE  DASUM
!***PURPOSE  Compute the sum of the magnitudes of the elements of a
!            vector.
!***LIBRARY   SLATEC (BLAS)
!***CATEGORY  D1A3A
!***TYPE      DOUBLE PRECISION (SASUM-S, DASUM-D, SCASUM-C)
!***KEYWORDS  BLAS, LINEAR ALGEBRA, SUM OF MAGNITUDES OF A VECTOR
!***AUTHOR  Lawson, C. L., (JPL)
!           Hanson, R. J., (SNLA)
!           Kincaid, D. R., (U. of Texas)
!           Krogh, F. T., (JPL)
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       DX  double precision vector with N elements
!     INCX  storage spacing between elements of DX
!
!     --Output--
!    DASUM  double precision result (zero if N .LE. 0)
!
!     Returns sum of magnitudes of double precision DX.
!     DASUM = sum from 0 to N-1 of ABS(DX(IX+I*INCX)),
!     where IX = 1 if INCX .GE. 0, else IX = 1+(1-N)*INCX.
!
!***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
!                 Krogh, Basic linear algebra subprograms for Fortran
!                 usage, Algorithm No. 539, Transactions on Mathematical
!                 Software 5, 3 (September 1979), pp. 308-323.
!***ROUTINES CALLED  (NONE)
!***REVISION HISTORY  (YYMMDD)
!   791001  DATE WRITTEN
!   890531  Changed all specific intrinsics to generic.  (WRB)
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   900821  Modified to correct problem with a negative increment.
!           (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DASUM
      DOUBLE PRECISION DX(*)
      INTEGER I, INCX, IX, M, MP1, N
!***FIRST EXECUTABLE STATEMENT  DASUM
      DASUM = 0.0D0
      IF (N .LE. 0) RETURN
!
      IF (INCX .EQ. 1) GOTO 20
!
!     Code for increment not equal to 1.
!
      IX = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      DO 10 I = 1,N
        DASUM = DASUM + ABS(DX(IX))
        IX = IX + INCX
   10 CONTINUE
      RETURN
!
!     Code for increment equal to 1.
!
!     Clean-up loop so remaining vector length is a multiple of 6.
!
   20 M = MOD(N,6)
      IF (M .EQ. 0) GOTO 40
      DO 30 I = 1,M
        DASUM = DASUM + ABS(DX(I))
   30 CONTINUE
      IF (N .LT. 6) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,6
        DASUM = DASUM + ABS(DX(I)) + ABS(DX(I+1)) + ABS(DX(I+2)) +
     1          ABS(DX(I+3)) + ABS(DX(I+4)) + ABS(DX(I+5))
   50 CONTINUE
      RETURN
      END
*DECK DAXPY
      SUBROUTINE DAXPY (N, DA, DX, INCX, DY, INCY)
!***BEGIN PROLOGUE  DAXPY
!***PURPOSE  Compute a constant times a vector plus a vector.
!***LIBRARY   SLATEC (BLAS)
!***CATEGORY  D1A7
!***TYPE      DOUBLE PRECISION (SAXPY-S, DAXPY-D, CAXPY-C)
!***KEYWORDS  BLAS, LINEAR ALGEBRA, TRIAD, VECTOR
!***AUTHOR  Lawson, C. L., (JPL)
!           Hanson, R. J., (SNLA)
!           Kincaid, D. R., (U. of Texas)
!           Krogh, F. T., (JPL)
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       DA  double precision scalar multiplier
!       DX  double precision vector with N elements
!     INCX  storage spacing between elements of DX
!       DY  double precision vector with N elements
!     INCY  storage spacing between elements of DY
!
!     --Output--
!       DY  double precision result (unchanged if N .LE. 0)
!
!     Overwrite double precision DY with double precision DA*DX + DY.
!     For I = 0 to N-1, replace  DY(LY+I*INCY) with DA*DX(LX+I*INCX) +
!       DY(LY+I*INCY),
!     where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
!     defined in a similar way using INCY.
!
!***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
!                 Krogh, Basic linear algebra subprograms for Fortran
!                 usage, Algorithm No. 539, Transactions on Mathematical
!                 Software 5, 3 (September 1979), pp. 308-323.
!***ROUTINES CALLED  (NONE)
!***REVISION HISTORY  (YYMMDD)
!   791001  DATE WRITTEN
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   920310  Corrected definition of LX in DESCRIPTION.  (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DAXPY
      DOUBLE PRECISION DX(*), DY(*), DA
!***FIRST EXECUTABLE STATEMENT  DAXPY
      IF (N.LE.0 .OR. DA.EQ.0.0D0) RETURN
      IF (INCX .EQ. INCY) IF (INCX-1) 5,20,60
!
!     Code for unequal or nonpositive increments.
!
    5 IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DY(IY) = DY(IY) + DA*DX(IX)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
!
!     Code for both increments equal to 1.
!
!     Clean-up loop so remaining vector length is a multiple of 4.
!
   20 M = MOD(N,4)
      IF (M .EQ. 0) GO TO 40
      DO 30 I = 1,M
        DY(I) = DY(I) + DA*DX(I)
   30 CONTINUE
      IF (N .LT. 4) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,4
        DY(I) = DY(I) + DA*DX(I)
        DY(I+1) = DY(I+1) + DA*DX(I+1)
        DY(I+2) = DY(I+2) + DA*DX(I+2)
        DY(I+3) = DY(I+3) + DA*DX(I+3)
   50 CONTINUE
      RETURN
!
!     Code for equal, positive, non-unit increments.
!
   60 NS = N*INCX
      DO 70 I = 1,NS,INCX
        DY(I) = DA*DX(I) + DY(I)
   70 CONTINUE
      RETURN
      END
*DECK DDOT
      DOUBLE PRECISION FUNCTION DDOT (N, DX, INCX, DY, INCY)
!***BEGIN PROLOGUE  DDOT
!***PURPOSE  Compute the inner product of two vectors.
!***LIBRARY   SLATEC (BLAS)
!***CATEGORY  D1A4
!***TYPE      DOUBLE PRECISION (SDOT-S, DDOT-D, CDOTU-C)
!***KEYWORDS  BLAS, INNER PRODUCT, LINEAR ALGEBRA, VECTOR
!***AUTHOR  Lawson, C. L., (JPL)
!           Hanson, R. J., (SNLA)
!           Kincaid, D. R., (U. of Texas)
!           Krogh, F. T., (JPL)
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       DX  double precision vector with N elements
!     INCX  storage spacing between elements of DX
!       DY  double precision vector with N elements
!     INCY  storage spacing between elements of DY
!
!     --Output--
!     DDOT  double precision dot product (zero if N .LE. 0)
!
!     Returns the dot product of double precision DX and DY.
!     DDOT = sum for I = 0 to N-1 of  DX(LX+I*INCX) * DY(LY+I*INCY),
!     where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
!     defined in a similar way using INCY.
!
!***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
!                 Krogh, Basic linear algebra subprograms for Fortran
!                 usage, Algorithm No. 539, Transactions on Mathematical
!                 Software 5, 3 (September 1979), pp. 308-323.
!***ROUTINES CALLED  (NONE)
!***REVISION HISTORY  (YYMMDD)
!   791001  DATE WRITTEN
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   920310  Corrected definition of LX in DESCRIPTION.  (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DDOT
      DOUBLE PRECISION DX(*), DY(*)
!***FIRST EXECUTABLE STATEMENT  DDOT
      DDOT = 0.0D0
      IF (N .LE. 0) RETURN
      IF (INCX .EQ. INCY) IF (INCX-1) 5,20,60
!
!     Code for unequal or nonpositive increments.
!
    5 IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DDOT = DDOT + DX(IX)*DY(IY)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
!
!     Code for both increments equal to 1.
!
!     Clean-up loop so remaining vector length is a multiple of 5.
!
   20 M = MOD(N,5)
      IF (M .EQ. 0) GO TO 40
      DO 30 I = 1,M
         DDOT = DDOT + DX(I)*DY(I)
   30 CONTINUE
      IF (N .LT. 5) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
      DDOT = DDOT + DX(I)*DY(I) + DX(I+1)*DY(I+1) + DX(I+2)*DY(I+2) +
     1              DX(I+3)*DY(I+3) + DX(I+4)*DY(I+4)
   50 CONTINUE
      RETURN
!
!     Code for equal, positive, non-unit increments.
!
   60 NS = N*INCX
      DO 70 I = 1,NS,INCX
        DDOT = DDOT + DX(I)*DY(I)
   70 CONTINUE
      RETURN
      END
*DECK DGECO
      SUBROUTINE DGECO (A, LDA, N, IPVT, RCOND, Z)
!***BEGIN PROLOGUE  DGECO
!***PURPOSE  Factor a matrix using Gaussian elimination and estimate
!            the condition number of the matrix.
!***LIBRARY   SLATEC (LINPACK)
!***CATEGORY  D2A1
!***TYPE      DOUBLE PRECISION (SGECO-S, DGECO-D, CGECO-C)
!***KEYWORDS  CONDITION NUMBER, GENERAL MATRIX, LINEAR ALGEBRA, LINPACK,
!             MATRIX FACTORIZATION
!***AUTHOR  Moler, C. B., (U. of New Mexico)
!***DESCRIPTION
!
!     DGECO factors a double precision matrix by Gaussian elimination
!     and estimates the condition of the matrix.
!
!     If  RCOND  is not needed, DGEFA is slightly faster.
!     To solve  A*X = B , follow DGECO by DGESL.
!     To compute  INVERSE(A)*C , follow DGECO by DGESL.
!     To compute  DETERMINANT(A) , follow DGECO by DGEDI.
!     To compute  INVERSE(A) , follow DGECO by DGEDI.
!
!     On Entry
!
!        A       DOUBLE PRECISION(LDA, N)
!                the matrix to be factored.
!
!        LDA     INTEGER
!                the leading dimension of the array  A .
!
!        N       INTEGER
!                the order of the matrix  A .
!
!     On Return
!
!        A       an upper triangular matrix and the multipliers
!                which were used to obtain it.
!                The factorization can be written  A = L*U  where
!                L  is a product of permutation and unit lower
!                triangular matrices and  U  is upper triangular.
!
!        IPVT    INTEGER(N)
!                an INTEGER vector of pivot indices.
!
!        RCOND   DOUBLE PRECISION
!                an estimate of the reciprocal condition of  A .
!                For the system  A*X = B , relative perturbations
!                in  A  and  B  of size  EPSILON  may cause
!                relative perturbations in  X  of size  EPSILON/RCOND .
!                If  RCOND  is so small that the logical expression
!                           1.0 + RCOND .EQ. 1.0
!                is true, then  A  may be singular to working
!                precision.  In particular,  RCOND  is zero  if
!                exact singularity is detected or the estimate
!                underflows.
!
!        Z       DOUBLE PRECISION(N)
!                a work vector whose contents are usually unimportant.
!                If  A  is close to a singular matrix, then  Z  is
!                an approximate null vector in the sense that
!                NORM(A*Z) = RCOND*NORM(A)*NORM(Z) .
!
!***REFERENCES  J. J. Dongarra, J. R. Bunch, C. B. Moler, and G. W.
!                 Stewart, LINPACK Users' Guide, SIAM, 1979.
!***ROUTINES CALLED  DASUM, DAXPY, DDOT, DGEFA, DSCAL
!***REVISION HISTORY  (YYMMDD)
!   780814  DATE WRITTEN
!   890531  Changed all specific intrinsics to generic.  (WRB)
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   900326  Removed duplicate information from DESCRIPTION section.
!           (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DGECO
      INTEGER LDA,N,IPVT(*)
      DOUBLE PRECISION A(LDA,*),Z(*)
      DOUBLE PRECISION RCOND
!
      DOUBLE PRECISION DDOT,EK,T,WK,WKM
      DOUBLE PRECISION ANORM,S,DASUM,SM,YNORM
      INTEGER INFO,J,K,KB,KP1,L
!
!     COMPUTE 1-NORM OF A
!
!***FIRST EXECUTABLE STATEMENT  DGECO
      ANORM = 0.0D0
      DO 10 J = 1, N
         ANORM = MAX(ANORM,DASUM(N,A(1,J),1))
   10 CONTINUE
!
!     FACTOR
!
      CALL DGEFA(A,LDA,N,IPVT,INFO)
!
!     RCOND = 1/(NORM(A)*(ESTIMATE OF NORM(INVERSE(A)))) .
!     ESTIMATE = NORM(Z)/NORM(Y) WHERE  A*Z = Y  AND  TRANS(A)*Y = E .
!     TRANS(A)  IS THE TRANSPOSE OF A .  THE COMPONENTS OF  E  ARE
!     CHOSEN TO CAUSE MAXIMUM LOCAL GROWTH IN THE ELEMENTS OF W  WHERE
!     TRANS(U)*W = E .  THE VECTORS ARE FREQUENTLY RESCALED TO AVOID
!     OVERFLOW.
!
!     SOLVE TRANS(U)*W = E
!
      EK = 1.0D0
      DO 20 J = 1, N
         Z(J) = 0.0D0
   20 CONTINUE
      DO 100 K = 1, N
         IF (Z(K) .NE. 0.0D0) EK = SIGN(EK,-Z(K))
         IF (ABS(EK-Z(K)) .LE. ABS(A(K,K))) GO TO 30
            S = ABS(A(K,K))/ABS(EK-Z(K))
            CALL DSCAL(N,S,Z,1)
            EK = S*EK
   30    CONTINUE
         WK = EK - Z(K)
         WKM = -EK - Z(K)
         S = ABS(WK)
         SM = ABS(WKM)
         IF (A(K,K) .EQ. 0.0D0) GO TO 40
            WK = WK/A(K,K)
            WKM = WKM/A(K,K)
         GO TO 50
   40    CONTINUE
            WK = 1.0D0
            WKM = 1.0D0
   50    CONTINUE
         KP1 = K + 1
         IF (KP1 .GT. N) GO TO 90
            DO 60 J = KP1, N
               SM = SM + ABS(Z(J)+WKM*A(K,J))
               Z(J) = Z(J) + WK*A(K,J)
               S = S + ABS(Z(J))
   60       CONTINUE
            IF (S .GE. SM) GO TO 80
               T = WKM - WK
               WK = WKM
               DO 70 J = KP1, N
                  Z(J) = Z(J) + T*A(K,J)
   70          CONTINUE
   80       CONTINUE
   90    CONTINUE
         Z(K) = WK
  100 CONTINUE
      S = 1.0D0/DASUM(N,Z,1)
      CALL DSCAL(N,S,Z,1)
!
!     SOLVE TRANS(L)*Y = W
!
      DO 120 KB = 1, N
         K = N + 1 - KB
         IF (K .LT. N) Z(K) = Z(K) + DDOT(N-K,A(K+1,K),1,Z(K+1),1)
         IF (ABS(Z(K)) .LE. 1.0D0) GO TO 110
            S = 1.0D0/ABS(Z(K))
            CALL DSCAL(N,S,Z,1)
  110    CONTINUE
         L = IPVT(K)
         T = Z(L)
         Z(L) = Z(K)
         Z(K) = T
  120 CONTINUE
      S = 1.0D0/DASUM(N,Z,1)
      CALL DSCAL(N,S,Z,1)
!
      YNORM = 1.0D0
!
!     SOLVE L*V = Y
!
      DO 140 K = 1, N
         L = IPVT(K)
         T = Z(L)
         Z(L) = Z(K)
         Z(K) = T
         IF (K .LT. N) CALL DAXPY(N-K,T,A(K+1,K),1,Z(K+1),1)
         IF (ABS(Z(K)) .LE. 1.0D0) GO TO 130
            S = 1.0D0/ABS(Z(K))
            CALL DSCAL(N,S,Z,1)
            YNORM = S*YNORM
  130    CONTINUE
  140 CONTINUE
      S = 1.0D0/DASUM(N,Z,1)
      CALL DSCAL(N,S,Z,1)
      YNORM = S*YNORM
!
!     SOLVE  U*Z = V
!
      DO 160 KB = 1, N
         K = N + 1 - KB
         IF (ABS(Z(K)) .LE. ABS(A(K,K))) GO TO 150
            S = ABS(A(K,K))/ABS(Z(K))
            CALL DSCAL(N,S,Z,1)
            YNORM = S*YNORM
  150    CONTINUE
         IF (A(K,K) .NE. 0.0D0) Z(K) = Z(K)/A(K,K)
         IF (A(K,K) .EQ. 0.0D0) Z(K) = 1.0D0
         T = -Z(K)
         CALL DAXPY(K-1,T,A(1,K),1,Z(1),1)
  160 CONTINUE
!     MAKE ZNORM = 1.0
      S = 1.0D0/DASUM(N,Z,1)
      CALL DSCAL(N,S,Z,1)
      YNORM = S*YNORM
!
      IF (ANORM .NE. 0.0D0) RCOND = YNORM/ANORM
      IF (ANORM .EQ. 0.0D0) RCOND = 0.0D0
      RETURN
      END
*DECK DGEDI
      SUBROUTINE DGEDI (A, LDA, N, IPVT, DET, WORK, JOB)
!***BEGIN PROLOGUE  DGEDI
!***PURPOSE  Compute the determinant and inverse of a matrix using the
!            factors computed by DGECO or DGEFA.
!***LIBRARY   SLATEC (LINPACK)
!***CATEGORY  D3A1, D2A1
!***TYPE      DOUBLE PRECISION (SGEDI-S, DGEDI-D, CGEDI-C)
!***KEYWORDS  DETERMINANT, INVERSE, LINEAR ALGEBRA, LINPACK, MATRIX
!***AUTHOR  Moler, C. B., (U. of New Mexico)
!***DESCRIPTION
!
!     DGEDI computes the determinant and inverse of a matrix
!     using the factors computed by DGECO or DGEFA.
!
!     On Entry
!
!        A       DOUBLE PRECISION(LDA, N)
!                the output from DGECO or DGEFA.
!
!        LDA     INTEGER
!                the leading dimension of the array  A .
!
!        N       INTEGER
!                the order of the matrix  A .
!
!        IPVT    INTEGER(N)
!                the pivot vector from DGECO or DGEFA.
!
!        WORK    DOUBLE PRECISION(N)
!                work vector.  Contents destroyed.
!
!        JOB     INTEGER
!                = 11   both determinant and inverse.
!                = 01   inverse only.
!                = 10   determinant only.
!
!     On Return
!
!        A       inverse of original matrix if requested.
!                Otherwise unchanged.
!
!        DET     DOUBLE PRECISION(2)
!                determinant of original matrix if requested.
!                Otherwise not referenced.
!                Determinant = DET(1) * 10.0**DET(2)
!                with  1.0 .LE. ABS(DET(1)) .LT. 10.0
!                or  DET(1) .EQ. 0.0 .
!
!     Error Condition
!
!        A division by zero will occur if the input factor contains
!        a zero on the diagonal and the inverse is requested.
!        It will not occur if the subroutines are called correctly
!        and if DGECO has set RCOND .GT. 0.0 or DGEFA has set
!        INFO .EQ. 0 .
!
!***REFERENCES  J. J. Dongarra, J. R. Bunch, C. B. Moler, and G. W.
!                 Stewart, LINPACK Users' Guide, SIAM, 1979.
!***ROUTINES CALLED  DAXPY, DSCAL, DSWAP
!***REVISION HISTORY  (YYMMDD)
!   780814  DATE WRITTEN
!   890531  Changed all specific intrinsics to generic.  (WRB)
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   900326  Removed duplicate information from DESCRIPTION section.
!           (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DGEDI
      INTEGER LDA,N,IPVT(*),JOB
      DOUBLE PRECISION A(LDA,*),DET(2),WORK(*)
!
      DOUBLE PRECISION T
      DOUBLE PRECISION TEN
      INTEGER I,J,K,KB,KP1,L,NM1
!***FIRST EXECUTABLE STATEMENT  DGEDI
!
!     COMPUTE DETERMINANT
!
      IF (JOB/10 .EQ. 0) GO TO 70
         DET(1) = 1.0D0
         DET(2) = 0.0D0
         TEN = 10.0D0
         DO 50 I = 1, N
            IF (IPVT(I) .NE. I) DET(1) = -DET(1)
            DET(1) = A(I,I)*DET(1)
            IF (DET(1) .EQ. 0.0D0) GO TO 60
   10       IF (ABS(DET(1)) .GE. 1.0D0) GO TO 20
               DET(1) = TEN*DET(1)
               DET(2) = DET(2) - 1.0D0
            GO TO 10
   20       CONTINUE
   30       IF (ABS(DET(1)) .LT. TEN) GO TO 40
               DET(1) = DET(1)/TEN
               DET(2) = DET(2) + 1.0D0
            GO TO 30
   40       CONTINUE
   50    CONTINUE
   60    CONTINUE
   70 CONTINUE
!
!     COMPUTE INVERSE(U)
!
      IF (MOD(JOB,10) .EQ. 0) GO TO 150
         DO 100 K = 1, N
            A(K,K) = 1.0D0/A(K,K)
            T = -A(K,K)
            CALL DSCAL(K-1,T,A(1,K),1)
            KP1 = K + 1
            IF (N .LT. KP1) GO TO 90
            DO 80 J = KP1, N
               T = A(K,J)
               A(K,J) = 0.0D0
               CALL DAXPY(K,T,A(1,K),1,A(1,J),1)
   80       CONTINUE
   90       CONTINUE
  100    CONTINUE
!
!        FORM INVERSE(U)*INVERSE(L)
!
         NM1 = N - 1
         IF (NM1 .LT. 1) GO TO 140
         DO 130 KB = 1, NM1
            K = N - KB
            KP1 = K + 1
            DO 110 I = KP1, N
               WORK(I) = A(I,K)
               A(I,K) = 0.0D0
  110       CONTINUE
            DO 120 J = KP1, N
               T = WORK(J)
               CALL DAXPY(N,T,A(1,J),1,A(1,K),1)
  120       CONTINUE
            L = IPVT(K)
            IF (L .NE. K) CALL DSWAP(N,A(1,K),1,A(1,L),1)
  130    CONTINUE
  140    CONTINUE
  150 CONTINUE
      RETURN
      END
*DECK DGEFA
      SUBROUTINE DGEFA (A, LDA, N, IPVT, INFO)
!***BEGIN PROLOGUE  DGEFA
!***PURPOSE  Factor a matrix using Gaussian elimination.
!***LIBRARY   SLATEC (LINPACK)
!***CATEGORY  D2A1
!***TYPE      DOUBLE PRECISION (SGEFA-S, DGEFA-D, CGEFA-C)
!***KEYWORDS  GENERAL MATRIX, LINEAR ALGEBRA, LINPACK,
!             MATRIX FACTORIZATION
!***AUTHOR  Moler, C. B., (U. of New Mexico)
!***DESCRIPTION
!
!     DGEFA factors a double precision matrix by Gaussian elimination.
!
!     DGEFA is usually called by DGECO, but it can be called
!     directly with a saving in time if  RCOND  is not needed.
!     (Time for DGECO) = (1 + 9/N)*(Time for DGEFA) .
!
!     On Entry
!
!        A       DOUBLE PRECISION(LDA, N)
!                the matrix to be factored.
!
!        LDA     INTEGER
!                the leading dimension of the array  A .
!
!        N       INTEGER
!                the order of the matrix  A .
!
!     On Return
!
!        A       an upper triangular matrix and the multipliers
!                which were used to obtain it.
!                The factorization can be written  A = L*U  where
!                L  is a product of permutation and unit lower
!                triangular matrices and  U  is upper triangular.
!
!        IPVT    INTEGER(N)
!                an integer vector of pivot indices.
!
!        INFO    INTEGER
!                = 0  normal value.
!                = K  if  U(K,K) .EQ. 0.0 .  This is not an error
!                     condition for this subroutine, but it does
!                     indicate that DGESL or DGEDI will divide by zero
!                     if called.  Use  RCOND  in DGECO for a reliable
!                     indication of singularity.
!
!***REFERENCES  J. J. Dongarra, J. R. Bunch, C. B. Moler, and G. W.
!                 Stewart, LINPACK Users' Guide, SIAM, 1979.
!***ROUTINES CALLED  DAXPY, DSCAL, IDAMAX
!***REVISION HISTORY  (YYMMDD)
!   780814  DATE WRITTEN
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   900326  Removed duplicate information from DESCRIPTION section.
!           (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DGEFA
      INTEGER LDA,N,IPVT(*),INFO
      DOUBLE PRECISION A(LDA,*)
!
      DOUBLE PRECISION T
      INTEGER IDAMAX,J,K,KP1,L,NM1
!
!     GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING
!
!***FIRST EXECUTABLE STATEMENT  DGEFA
      INFO = 0
      NM1 = N - 1
      IF (NM1 .LT. 1) GO TO 70
      DO 60 K = 1, NM1
         KP1 = K + 1
!
!        FIND L = PIVOT INDEX
!
         L = IDAMAX(N-K+1,A(K,K),1) + K - 1
         IPVT(K) = L
!
!        ZERO PIVOT IMPLIES THIS COLUMN ALREADY TRIANGULARIZED
!
         IF (A(L,K) .EQ. 0.0D0) GO TO 40
!
!           INTERCHANGE IF NECESSARY
!
            IF (L .EQ. K) GO TO 10
               T = A(L,K)
               A(L,K) = A(K,K)
               A(K,K) = T
   10       CONTINUE
!
!           COMPUTE MULTIPLIERS
!
            T = -1.0D0/A(K,K)
            CALL DSCAL(N-K,T,A(K+1,K),1)
!
!           ROW ELIMINATION WITH COLUMN INDEXING
!
            DO 30 J = KP1, N
               T = A(L,J)
               IF (L .EQ. K) GO TO 20
                  A(L,J) = A(K,J)
                  A(K,J) = T
   20          CONTINUE
               CALL DAXPY(N-K,T,A(K+1,K),1,A(K+1,J),1)
   30       CONTINUE
         GO TO 50
   40    CONTINUE
            INFO = K
   50    CONTINUE
   60 CONTINUE
   70 CONTINUE
      IPVT(N) = N
      IF (A(N,N) .EQ. 0.0D0) INFO = N
      RETURN
      END
*DECK DSCAL
      SUBROUTINE DSCAL (N, DA, DX, INCX)
!***BEGIN PROLOGUE  DSCAL
!***PURPOSE  Multiply a vector by a constant.
!***LIBRARY   SLATEC (BLAS)
!***CATEGORY  D1A6
!***TYPE      DOUBLE PRECISION (SSCAL-S, DSCAL-D, CSCAL-C)
!***KEYWORDS  BLAS, LINEAR ALGEBRA, SCALE, VECTOR
!***AUTHOR  Lawson, C. L., (JPL)
!           Hanson, R. J., (SNLA)
!           Kincaid, D. R., (U. of Texas)
!           Krogh, F. T., (JPL)
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       DA  double precision scale factor
!       DX  double precision vector with N elements
!     INCX  storage spacing between elements of DX
!
!     --Output--
!       DX  double precision result (unchanged if N.LE.0)
!
!     Replace double precision DX by double precision DA*DX.
!     For I = 0 to N-1, replace DX(IX+I*INCX) with  DA * DX(IX+I*INCX),
!     where IX = 1 if INCX .GE. 0, else IX = 1+(1-N)*INCX.
!
!***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
!                 Krogh, Basic linear algebra subprograms for Fortran
!                 usage, Algorithm No. 539, Transactions on Mathematical
!                 Software 5, 3 (September 1979), pp. 308-323.
!***ROUTINES CALLED  (NONE)
!***REVISION HISTORY  (YYMMDD)
!   791001  DATE WRITTEN
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   900821  Modified to correct problem with a negative increment.
!           (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DSCAL
      DOUBLE PRECISION DA, DX(*)
      INTEGER I, INCX, IX, M, MP1, N
!***FIRST EXECUTABLE STATEMENT  DSCAL
      IF (N .LE. 0) RETURN
      IF (INCX .EQ. 1) GOTO 20
!
!     Code for increment not equal to 1.
!
      IX = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      DO 10 I = 1,N
        DX(IX) = DA*DX(IX)
        IX = IX + INCX
   10 CONTINUE
      RETURN
!
!     Code for increment equal to 1.
!
!     Clean-up loop so remaining vector length is a multiple of 5.
!
   20 M = MOD(N,5)
      IF (M .EQ. 0) GOTO 40
      DO 30 I = 1,M
        DX(I) = DA*DX(I)
   30 CONTINUE
      IF (N .LT. 5) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
        DX(I) = DA*DX(I)
        DX(I+1) = DA*DX(I+1)
        DX(I+2) = DA*DX(I+2)
        DX(I+3) = DA*DX(I+3)
        DX(I+4) = DA*DX(I+4)
   50 CONTINUE
      RETURN
      END
*DECK DSWAP
      SUBROUTINE DSWAP (N, DX, INCX, DY, INCY)
!***BEGIN PROLOGUE  DSWAP
!***PURPOSE  Interchange two vectors.
!***LIBRARY   SLATEC (BLAS)
!***CATEGORY  D1A5
!***TYPE      DOUBLE PRECISION (SSWAP-S, DSWAP-D, CSWAP-C, ISWAP-I)
!***KEYWORDS  BLAS, INTERCHANGE, LINEAR ALGEBRA, VECTOR
!***AUTHOR  Lawson, C. L., (JPL)
!           Hanson, R. J., (SNLA)
!           Kincaid, D. R., (U. of Texas)
!           Krogh, F. T., (JPL)
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       DX  double precision vector with N elements
!     INCX  storage spacing between elements of DX
!       DY  double precision vector with N elements
!     INCY  storage spacing between elements of DY
!
!     --Output--
!       DX  input vector DY (unchanged if N .LE. 0)
!       DY  input vector DX (unchanged if N .LE. 0)
!
!     Interchange double precision DX and double precision DY.
!     For I = 0 to N-1, interchange  DX(LX+I*INCX) and DY(LY+I*INCY),
!     where LX = 1 if INCX .GE. 0, else LX = 1+(1-N)*INCX, and LY is
!     defined in a similar way using INCY.
!
!***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
!                 Krogh, Basic linear algebra subprograms for Fortran
!                 usage, Algorithm No. 539, Transactions on Mathematical
!                 Software 5, 3 (September 1979), pp. 308-323.
!***ROUTINES CALLED  (NONE)
!***REVISION HISTORY  (YYMMDD)
!   791001  DATE WRITTEN
!   890831  Modified array declarations.  (WRB)
!   890831  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   920310  Corrected definition of LX in DESCRIPTION.  (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  DSWAP
      DOUBLE PRECISION DX(*), DY(*), DTEMP1, DTEMP2, DTEMP3
!***FIRST EXECUTABLE STATEMENT  DSWAP
      IF (N .LE. 0) RETURN
      IF (INCX .EQ. INCY) IF (INCX-1) 5,20,60
!
!     Code for unequal or nonpositive increments.
!
    5 IX = 1
      IY = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      IF (INCY .LT. 0) IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DTEMP1 = DX(IX)
        DX(IX) = DY(IY)
        DY(IY) = DTEMP1
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
!
!     Code for both increments equal to 1.
!
!     Clean-up loop so remaining vector length is a multiple of 3.
!
   20 M = MOD(N,3)
      IF (M .EQ. 0) GO TO 40
      DO 30 I = 1,M
        DTEMP1 = DX(I)
        DX(I) = DY(I)
        DY(I) = DTEMP1
   30 CONTINUE
      IF (N .LT. 3) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,3
        DTEMP1 = DX(I)
        DTEMP2 = DX(I+1)
        DTEMP3 = DX(I+2)
        DX(I) = DY(I)
        DX(I+1) = DY(I+1)
        DX(I+2) = DY(I+2)
        DY(I) = DTEMP1
        DY(I+1) = DTEMP2
        DY(I+2) = DTEMP3
   50 CONTINUE
      RETURN
!
!     Code for equal, positive, non-unit increments.
!
   60 NS = N*INCX
      DO 70 I = 1,NS,INCX
        DTEMP1 = DX(I)
        DX(I) = DY(I)
        DY(I) = DTEMP1
   70 CONTINUE
      RETURN
      END
*DECK IDAMAX
      INTEGER FUNCTION IDAMAX (N, DX, INCX)
!***BEGIN PROLOGUE  IDAMAX
!***PURPOSE  Find the smallest index of that component of a vector
!            having the maximum magnitude.
!***LIBRARY   SLATEC (BLAS)
!***CATEGORY  D1A2
!***TYPE      DOUBLE PRECISION (ISAMAX-S, IDAMAX-D, ICAMAX-C)
!***KEYWORDS  BLAS, LINEAR ALGEBRA, MAXIMUM COMPONENT, VECTOR
!***AUTHOR  Lawson, C. L., (JPL)
!           Hanson, R. J., (SNLA)
!           Kincaid, D. R., (U. of Texas)
!           Krogh, F. T., (JPL)
!***DESCRIPTION
!
!                B L A S  Subprogram
!    Description of Parameters
!
!     --Input--
!        N  number of elements in input vector(s)
!       DX  double precision vector with N elements
!     INCX  storage spacing between elements of DX
!
!     --Output--
!   IDAMAX  smallest index (zero if N .LE. 0)
!
!     Find smallest index of maximum magnitude of double precision DX.
!     IDAMAX = first I, I = 1 to N, to maximize ABS(DX(IX+(I-1)*INCX)),
!     where IX = 1 if INCX .GE. 0, else IX = 1+(1-N)*INCX.
!
!***REFERENCES  C. L. Lawson, R. J. Hanson, D. R. Kincaid and F. T.
!                 Krogh, Basic linear algebra subprograms for Fortran
!                 usage, Algorithm No. 539, Transactions on Mathematical
!                 Software 5, 3 (September 1979), pp. 308-323.
!***ROUTINES CALLED  (NONE)
!***REVISION HISTORY  (YYMMDD)
!   791001  DATE WRITTEN
!   890531  Changed all specific intrinsics to generic.  (WRB)
!   890531  REVISION DATE from Version 3.2
!   891214  Prologue converted to Version 4.0 format.  (BAB)
!   900821  Modified to correct problem with a negative increment.
!           (WRB)
!   920501  Reformatted the REFERENCES section.  (WRB)
!***END PROLOGUE  IDAMAX
      DOUBLE PRECISION DX(*), DMAX, XMAG
      INTEGER I, INCX, IX, N
!***FIRST EXECUTABLE STATEMENT  IDAMAX
      IDAMAX = 0
      IF (N .LE. 0) RETURN
      IDAMAX = 1
      IF (N .EQ. 1) RETURN
!
      IF (INCX .EQ. 1) GOTO 20
!
!     Code for increments not equal to 1.
!
      IX = 1
      IF (INCX .LT. 0) IX = (-N+1)*INCX + 1
      DMAX = ABS(DX(IX))
      IX = IX + INCX
      DO 10 I = 2,N
        XMAG = ABS(DX(IX))
        IF (XMAG .GT. DMAX) THEN
          IDAMAX = I
          DMAX = XMAG
        ENDIF
        IX = IX + INCX
   10 CONTINUE
      RETURN
!
!     Code for increments equal to 1.
!
   20 DMAX = ABS(DX(1))
      DO 30 I = 2,N
        XMAG = ABS(DX(I))
        IF (XMAG .GT. DMAX) THEN
          IDAMAX = I
          DMAX = XMAG
        ENDIF
   30 CONTINUE
      RETURN
      END

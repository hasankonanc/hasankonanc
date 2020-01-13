clear all;clc;close all

AnalogvsDigital={'Analog Modulations';'Dijital Modulations'};
AnalogvsDigitalChoose=menu('Analog vs Digital',AnalogvsDigital(1),AnalogvsDigital(2));

if AnalogvsDigitalChoose==1
ModulationAnalog={'AM';'DSB';'SSB';'FM'};
ModulasyonAna = menu('Please choose modulation types.', ModulationAnalog(1), ModulationAnalog(2), ModulationAnalog(3), ModulationAnalog(4));
 
if ModulasyonAna == 1
    
clc; clear all; close all;
     
a  = input('Message signal amplitude valuable: a= ');
fm = input('Message signal frequency valuable: fm= ');
A  = input('Carrier signal amplitude valuable: A= ');
fc = input('Carrier signal frequency valuable(fc>>fm): fc= ');
ma = input('Modulation index: ma= ');

t=0:0.001:1;

m_t = a*cos(2*pi*fm*t);
c_t = A*cos(2*pi*fc*t);
u_t = (1+ma*m_t).*c_t; 

subplot(3,1,1);
plot (t,m_t,'b');
title('Message Signal')

subplot(3,1,2);
plot (t,c_t,'r');
title('Carrier Signal')

subplot(3,1,3);
plot (t,u_t,'k');
title('AM Signal') 

elseif ModulasyonAna ==2
    
clc; clear all; close all;

a  = input('Message signal amplitude valuable: a= ');
fm = input('Message signal frequency valuable: fm= ');
A  = input('Carrier signal amplitude valuable: A= ');
fc = input('Carrier signal frequency valuable:(fc>>fm): fc= ');

t=0:0.001:1;
Fs=1000;
m_t   = a*cos(2*pi*fm*t);
c_t = A*cos(2*pi*fc*t);
output = (m_t.*c_t);

figure,
subplot(311),plot(t,m_t),title('..........'),grid on
xlabel('Time [s]'),ylabel('Amplitude [V]')
subplot(312),plot(t,c_t),title('..........'),grid on
xlabel('Time [s]'),ylabel('Amplitude [V]')
subplot(313),plot(t,output),title('..........'),grid on
xlabel('Time [s]'),ylabel('Amplitude [V]')

M = fftshift(abs(fft(m_t)));
S = fftshift(abs(fft(output)));
F = linspace(-Fs/2 , Fs/2 , numel(M));
figure
plot(F,S,'b')

elseif ModulasyonAna ==3
      
clc; clear all; close all;
    
a  = input('Message signal amplitude valuable: a= ');
fm = input('Message signal frequency valuable: fm= ');
A  = input('Carrier signal amplitude valuable: A= ');
fc = input('Carrier signal frequency valuable:(fc>>fm): fc= ');

t=0:0.001:1;

m  = a*cos(2*pi*fm*t);
c1 = A*cos(2*pi*fc*t);
c2 = A*sin(2*pi*fc*t);
y = imag(hilbert(m));
u1  = (m.*c1)-(y.*c2) ; % Upper Side Band
u2  = (m.*c1)+(y.*c2) ; % Lower Side Band

subplot(6,1,1);
plot (t,m,'b');
title('Message Signal')

subplot(6,1,2);
plot (t,c1,'r');
title('1. Carrier Signal')

subplot(6,1,3);
plot (t,c2,'k');
title('2. Carrier Signal')

subplot(6,1,4);
plot (t,y,'k');
title('Hilbert')

subplot(6,1,5);
plot (t,u1,'g','linewidth',1.5);
title('Upper Side Band Signal')
hold on
subplot(6,1,6);
plot (t,u1,'r');
title('Lower Side Band Signal')
 
elseif ModulasyonAna ==4
    
clc; clear all; close all;

t = 0:0.001:1;

a=input('Message signal amplitude valuable: a =  ');
fm=input('Message signal frequency valuable: fm = '); 
A=input('Carrier signal amplitude valuable: A = ');
fc=input('Carrier signal frequency valuable:(fc>>fm): fc = ');
mf = input('Modulation index:  mf= ');

m = a*cos(2*pi*fm*t);
subplot(3,1,1); 
plot(t,m,'b');
title('Message Signal');

c = A*cos(2*pi*fc*t);
subplot(3,1,2); 
plot(t,c,'r');
title('Carrier Signal');

y = A*cos(2*pi*fc*t+mf.*sin(2*pi*fm*t));
subplot(3,1,3);
plot(t,y,'m','Linewidth',1.5);
title('FM Signal');
end

elseif AnalogvsDigitalChoose==2

ModulasyonDigital={'ASK';'FSK';'PSK';'QAM'};
ModulasyonDig = menu('Please Choose Modulation Types.',ModulasyonDigital(1),ModulasyonDigital(2),ModulasyonDigital(3),ModulasyonDigital(4));
    
 if ModulasyonDig ==1
 ASK={'2-ASK';'4-ASK';'16-ASK'};    
 ASK_Choose=menu('Please M-ASK Level.',ASK(1), ASK(2),ASK(3));
 
 if ASK_Choose==1
     
fc=70;
Q = 2; 
N = 20; 
b = randi([0,1],1,N);

t1_increase=0.0001; 
t1_final=1;

t1=0:t1_increase:(t1_final-t1_increase); 
t2=0:(t1_increase/2):(t1_final-t1_increase/2); 

w=(t1_final/t1_increase)/(N/2); 

carrier=cos(2*pi*fc*t1);

M = [];
for i= 2:2:length(b)  
        if  b(i)==0 
            bitler1(i/2)=0;
        elseif b(i)==1 
            bitler1(i/2)=1;

        end
        M = [M bitler1(i/2)];
end

q=[];                            
for i=1:(N/2)
    q = [q randi([M(i),M(i)],1,w)]; 
end

message=q./(Q-1); 
modulated=message.*carrier; 
figure
subplot(3,1,1)
plot(t1,message,':r','linewidth',3.5)
xlabel('Time');
ylabel('Amplitude');
title('Message Signal')
subplot(3,1,2) 
plot(t1,carrier,'b','linewidth',2.5)
xlabel('Time');
ylabel('Amplitude');
title('Carrier Signal')

subplot(3,1,3) 
plot(t1,modulated,'-k','linewidth',1.5)
xlabel('Time');
ylabel('Amplitude');
title('Modulated Signal')



 elseif ASK_Choose ==2

%4 ASK
fc=70;
Q2 = 4; 
N = 20;
b = randi([0,1],1,N); 

t1_increase=0.0001; 
t1_final=1;

t1=0:t1_increase:(t1_final-t1_increase); %tasiyici frekans icin t grafiginin matris degerlerini verir

w=(t1_final/t1_increase)/(N/2); %paketlerde kacar tane deger olacagini ifade eder

carrier=cos(2*pi*fc*t1); %taþýyýcý iþaretin matrisini ifade eder

M = [];
for i= 2:2:length(b) % elimizdeki 0 ve 1 deðerlerini modülasyon seviyesi kadar farklý eþit aralýklý 
        if b(i-1)==0 && b(i)==0 % deðerlere çevirir
            bitler1(i/2)=0;
        elseif b(i-1)==0 &&  b(i)==1 
            bitler1(i/2)=1;
        elseif b(i-1)==1 &&  b(i)==0 
             bitler1(i/2)=2;
        elseif b(i-1)==1 &&  b(i)==1 
             bitler1(i/2)=3;
        end
        M = [M bitler1(i/2)];
end

q=[];                               %farklý modülasyon seviyesi deðerlerini paketlere çevirir
for i=1:(N/2)
    q = [q randi([M(i),M(i)],1,w)]; 
end

message=q./(Q2-1); 
modulated=message.*carrier; 
figure
subplot(3,1,1)
plot(t1,message,':r','linewidth',3.5)
xlabel('Time');
ylabel('Amplitude');
title('Message Signal')
subplot(3,1,2) 
plot(t1,carrier,'b','linewidth',2.5)
xlabel('Time');
ylabel('Amplitude');
title('Carrier Signal')
subplot(3,1,3) 
plot(t1,modulated,'-k','linewidth',1.5)
xlabel('Time');
ylabel('Amplitude');
title('Modulated Signal')


elseif ASK_Choose ==3
fc=70;

Q = 16; 
N = 40; 
b = randi([0,1],1,N);

t1_increase=0.0001;
t1_final=1; 

t1=0:t1_increase:(t1_final-t1_increase);

w=(t1_final/t1_increase)/(N/4);

carrier=cos(2*pi*fc*t1);

M = [];
for i= 4:4:length(b)
        if b(i-3)==0 && b(i-2)==0 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=0;
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==0 && b(i)==1 
            bitler1(i/4)=1;
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==1 && b(i)==0 
            bitler1(i/4)=2;
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=3;
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=4;
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==0 && b(i)==1 
            bitler1(i/4)=5;
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=6;
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=7;
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=8;
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=9;
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=10;
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=11;
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==0 && b(i)==0 
            bitler1(i/4)=12;
         elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=13;
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=14;
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=15;
            
        end
        M = [M bitler1(i/4)];
end

q=[];  
for i=1:(N/4)
    q = [q randi([M(i),M(i)],1,w)]; 
end

g_bit=[];

for i= 1:length(q)
        if q(i)==0 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=0;
            bitler2(4*i)=0;
        elseif q(i)==1 
             bitler2(4*i-3)=0;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=0;
            bitler2(4*i)=1;
        elseif q(i)==2 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=1;
            bitler2(4*i)=0;
        elseif q(i)==3 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=1;
            bitler2(4*i)=1;
            elseif q(i)==4 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=0;
            bitler2(4*i)=0;
            elseif q(i)==5 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=0;
            bitler2(4*i)=1;
            elseif q(i)==6 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=1;
            bitler2(4*i)=0;
            elseif q(i)==7 
            bitler2(4*i-3)=0;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=1;
            bitler2(4*i)=1;
            elseif q(i)==8 
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=0;
            bitler2(4*i)=0;
            elseif q(i)==9 
           bitler2(4*i-3)=1;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=0;
            bitler2(4*i)=1;
            elseif q(i)==10 
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=1;
            bitler2(4*i)=0;
            elseif q(i)==11 
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=0;
            bitler2(4*i-1)=1;
            bitler2(4*i)=1;
            elseif q(i)==12
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=0;
            bitler2(4*i)=0;
            elseif q(i)==13 
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=0;
            bitler2(4*i)=1;
            elseif q(i)==14 
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=1;
            bitler2(4*i)=0;
            elseif q(i)==15 
            bitler2(4*i-3)=1;
            bitler2(4*i-2)=1;
            bitler2(4*i-1)=1;
            bitler2(4*i)=1;
        end       
end

g_bit=[ g_bit bitler2 ];

message=q./(Q-1); 
modulated=message.*carrier; %

subplot(3,1,1) 
plot(t1,message,':r','linewidth',3.5)
xlabel('Time');
ylabel('Amplitude');
title('Message Signal')
subplot(3,1,2) %ekrana taþýyýcý iþareti çizdirir
plot(t1,carrier,'b','linewidth',2.5)
xlabel('Time');
ylabel('Amplitude');
title('Carrier Signal')
subplot(3,1,3) 
plot(t1,modulated,'-k','linewidth',1.5) %ekrana modüleli iþareti çizdirir
xlabel('Time');
ylabel('Amplitude');
title('Modulated Signal')
 
 end
 
elseif ModulasyonDig ==2
    
FSK={'BFSK';'QFSK'};    
FSK_Choose=menu('Please Choose M-FSK Modulation Level .',FSK(1), FSK(2));

if FSK_Choose==1
    
 f=1; 
 f2=3; 
 f3=7; 
 f4=9; 
 fs=1000; 
 ts=1/fs; 
N=50;
b = randi([0,1],1,N);
bits = [];
for i= 1:1:length(b)  
        if b(i)==0
            bitler1(i)=0;
        elseif b(i)==1 
            bitler1(i)=1;
        end
        bits = [bits bitler1(i)];
end

 n_bits=numel(bits); 
 FSK=zeros(fs,n_bits); 
 
 for i=1:n_bits 
 t = i-1:ts:i-ts; 
 
 if bits(i)==0  
         fsk=cos(2*pi*f*t);
 
 elseif bits(i)==1 
         fsk=cos(2*pi*f2*t);
 

 end
         
 FSK(:,i)=fsk';
 plot(t,fsk);
 hold on;
 axis([0 n_bits -4 4]);
 end
 FSK=FSK(:)'; 
 xlabel('Time')
 ylabel('Amplitude')
 title('2FSK')
 
elseif FSK_Choose==2
    
 f=1; 
 f2=3; 
 f3=7; 
 f4=9; 
 fs=1000; 
 ts=1/fs; 
N=20;
b = randi([0,1],1,N);
bits = [];
for i= 2:2:length(b) 
        if b(i-1)==0 && b(i)==0
            bitler1(i/2)=0;
        elseif b(i-1)==0 && b(i)==1
            bitler1(i/2)=1;
        elseif b(i-1)==1 && b(i)==0
            bitler1(i/2)=2;
        elseif b(i-1)==1 && b(i)==1
            bitler1(i/2)=3;
        end
        bits = [bits bitler1(i/2)];
end

 n_bits=numel(bits); 
 FSK=zeros(fs,n_bits); 
 
 for i=1:n_bits 
 t = i-1:ts:i-ts;
 
 if bits(i)==0 
         fsk=cos(2*pi*f*t);
 
 elseif bits(i)==1 
         fsk=cos(2*pi*f2*t);
 
 elseif bits(i)==2 
         fsk=cos(2*pi*f3*t);
         
 elseif bits(i)==3 
         fsk=cos(2*pi*f4*t);

 end
         
 FSK(:,i)=fsk';
 plot(t,fsk);
 hold on;
 axis([0 n_bits -4 4]);
 end
 FSK=FSK(:)';
 xlabel('Time')
 ylabel('Amplitude')
 title('4FSK')
end
 
elseif ModulasyonDig ==3

PSK={'BPSK';'QPSK';'8PSK';'16PSK'};    
PSK_Choose=menu('Please Choose M-PSK Modulation Level.',PSK(1), PSK(2),PSK(3),PSK(4));
 
if PSK_Choose==1
    
close;clear;clc;
f=5; 
phi=pi;
fs=1000; 
ts=1/fs; 
N=10;

b = randi([0,1],1,N);
bits = [];
for i= 1:1:length(b) 
        if b(i)==0
            bitler1(i)=0;
        elseif b(i)==1 
            bitler1(i)=1;
        end
        bits = [bits bitler1(i)];
end

n_bits=numel(bits); 
PSK=zeros(fs,n_bits); 
for i=1:n_bits 
t = i-1:ts:i-ts; 

if bits(i)==0
         k(i)=0;
         psk = cos(2*pi*f*t+k(i));
 
 elseif bits(i)==1
        k(i)=phi;
        psk = cos(2*pi*f*t+k(i));

 end

PSK(:,i)=psk';
plot(t,psk);
hold on;
grid on;
axis([0 n_bits -4 4]);
end
PSK=PSK(:)'; 
xlabel('Time')
ylabel('Amplitude')
title('BPSK');

thetas=exp(1i*k);
scatterplot(thetas)
title('BPSK Constellation Diagram')

elseif PSK_Choose==2
    
close;clear;clc;
f=5; 
phi=pi; 
fs=1000; 
ts=1/fs;
N=100;
b = randi([0,1],1,N);
bits = [];
for i= 2:2:length(b) 
        if b(i-1)==0 && b(i)==0 
            bitler1(i/2)=0;
        elseif b(i-1)==0 &&  b(i)==1 
            bitler1(i/2)=1;
        elseif b(i-1)==1 &&  b(i)==0 
             bitler1(i/2)=2;
        elseif b(i-1)==1 &&  b(i)==1 
             bitler1(i/2)=3;
        end
        bits = [bits bitler1(i/2)];
end

n_bits=numel(bits); 
PSK=zeros(fs,n_bits); 

for i=1:n_bits 
t = i-1:ts:i-ts; 

if bits(i)==0
         k(i)=+0.25*phi;
         psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==1
        k(i)=+0.75*phi;
        psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==2
         k(i)=-0.75*phi;
         psk = cos(2*pi*f*t+k(i));


 elseif  bits(i)==3
     k(i)=-0.25*phi;
     psk = cos(2*pi*f*t+k(i));

 end

PSK(:,i)=psk';
plot(t,psk);
hold on;
grid on;
axis([0 n_bits -4 4]);
end
PSK=PSK(:)'; 
xlabel('Time')
ylabel('Amplitude')
title('QPSK');

thetas=exp(1i*k);
scatterplot(thetas)
title('QPSK Constellation Diagram')

 elseif PSK_Choose==3
     
close;clear;clc;
f=5; 
phi=pi; 
phi2=pi/2;
phi3=pi+(pi/2);
fs=1000; 
ts=1/fs; 
N=48;
b = randi([0,1],1,N);
bits = [];
for i= 3:3:length(b) 
        if    b(i-2)==0 && b(i-1)==0 && b(i)==0 
            bitler1(i/3)=0;
        elseif b(i-2)==0 && b(i-1)==0 && b(i)==1 
            bitler1(i/3)=1;
        elseif b(i-2)==0 && b(i-1)==1 && b(i)==1
             bitler1(i/3)=2;
        elseif b(i-2)==0 && b(i-1)==1 && b(i)==0 
             bitler1(i/3)=3;
        elseif b(i-2)==1 && b(i-1)==1 && b(i)==0 
             bitler1(i/3)=4;
        elseif b(i-2)==1 && b(i-1)==1 && b(i)==1 
             bitler1(i/3)=5;
        elseif b(i-2)==1 && b(i-1)==0 && b(i)==1 
             bitler1(i/3)=6;
        elseif b(i-2)==1 && b(i-1)==0 && b(i)==0 
             bitler1(i/3)=7;
        end
        
        bits = [bits bitler1(i/3)];
end

n_bits=numel(bits); 
PSK=zeros(fs,n_bits); 
for i=1:n_bits 
t = i-1:ts:i-ts; 

 if      bits(i)==0
         k(i)=0;
         psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==1
        k(i)=+0.25*phi;
        psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==2
         k(i)=0.5*phi;
         psk = cos(2*pi*f*t+k(i));

 elseif  bits(i)==3
         k(i)=0.75*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==4
         k(i)= +phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==5
         k(i)=-0.25*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==6
         k(i)=-0.5*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==7
         k(i)=-0.75*phi;
         psk = cos(2*pi*f*t+k(i));
 end

PSK(:,i)=psk';
plot(t,psk);
hold on;
grid on;
axis([0 n_bits -4 4]);
end
PSK=PSK(:)'; 
xlabel('Time')
ylabel('Amplitude')
title('8-PSK');

thetas=exp(1i*k);
scatterplot(thetas)
title('8-PSK Constellation Diagram')

 elseif PSK_Choose==4
     
     
close;clear;clc;
f=5;
phi=pi; 
phi2=pi/2;
phi3=pi+(pi/2);
fs=1000; 
ts=1/fs;
N=256;

b = randi([0,1],1,N);
bits = [];
for i= 4:4:length(b)
        if b(i-3)==0 && b(i-2)==0 && b(i-1)==0 && b(i)==0
            bitler1(i/4)=0;
            
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=1;
            
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=2;
            
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=3;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=4;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=5;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=6;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=7;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=8;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==0 && b(i)==1 
            bitler1(i/4)=9;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=10;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=11;
            
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=12;
            
         elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=13;
            
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=14;
            
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=15;
            
        end
        bits = [bits bitler1(i/4)];
end

n_bits=numel(bits); 
PSK=zeros(fs,n_bits);
for i=1:n_bits 
t = i-1:ts:i-ts;

 if      bits(i)==0
         k(i)=0;
         psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==1
        k(i)=+0.125*phi;
        psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==2
         k(i)=0.25*phi;
         psk = cos(2*pi*f*t+k(i));

 elseif  bits(i)==3
         k(i)=0.375*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==4
         k(i)= 0.5*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==5
         k(i)= 0.625*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==6
         k(i)= 0.75*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==7
         k(i)= 0.875*phi;
         psk = cos(2*pi*f*t+k(i));
         
  elseif bits(i)==8
        k(i)=+phi;
        psk = cos(2*pi*f*t+k(i));

 
 elseif bits(i)==9
         k(i)=-0.125*phi;
         psk = cos(2*pi*f*t+k(i));

 elseif  bits(i)==10
         k(i)=-0.25*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==11
         k(i)= -0.375*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==12
         k(i)=-0.5*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==13
         k(i)=-0.625*phi;
         psk = cos(2*pi*f*t+k(i));
     
 elseif  bits(i)==14
         k(i)=-0.75*phi;
         psk = cos(2*pi*f*t+k(i));
         
 elseif  bits(i)==15
         k(i)=-0.875*phi;
         psk = cos(2*pi*f*t+k(i));
 end

PSK(:,i)=psk';
plot(t,psk);
hold on;
grid on;
axis([0 n_bits -4 4]);
end
PSK=PSK(:)'; 
xlabel('Time')
ylabel('Amplitude')
title('16-PSK');

thetas=exp(1i*k);
scatterplot(thetas)
title('16-PSK Constellation Diagram')
end

elseif ModulasyonDig ==4
     
QAM={'4-QAM';'16-QAM'};    
QAM_Choose=menu('Please Choose M-QAM Modulation Level.',QAM(1), QAM(2));

if QAM_Choose==1 
    
close;clear;clc;
f=5;
phi=pi; 
fs=1000;
ts=1/fs;
N=100;
b = randi([0,1],1,N);

bits = [];
for i= 2:2:length(b) 
        if b(i-1)==0 && b(i)==0 
            bitler1(i/2)=0;
        elseif b(i-1)==0 &&  b(i)==1 
            bitler1(i/2)=1;
        elseif b(i-1)==1 &&  b(i)==0 
             bitler1(i/2)=2;
        elseif b(i-1)==1 &&  b(i)==1 
             bitler1(i/2)=3;
        end
        bits = [bits bitler1(i/2)];
end

n_bits=numel(bits); 
QAM=zeros(fs,n_bits); 

for i=1:n_bits 
t = i-1:ts:i-ts; 

if bits(i)==0
         k(i)=+0.25*phi;
         qam=exp(1j*k(i));

 
 elseif bits(i)==1
        k(i)=+0.75*phi;
        qam=exp(1j*k(i));
 
 elseif bits(i)==2
        k(i)=-0.75*phi;
        qam=exp(1j*k(i));

 elseif  bits(i)==3
        k(i)=-0.25*phi;
        qam=exp(1j*k(i));
        
end

QAM(:,i)=qam';
end
QAM=QAM(:)';
thetas= 1.4142*exp(1i*k);

plot(thetas,'r.')
title('QAM Constellation Diagram')

elseif QAM_Choose==2

close;clear;clc;
f=5; 
phi=pi; 
fs=1000; 
ts=1/fs; 
N=300;
b = randi([0,1],1,N);

bits = [];
for i= 4:4:length(b) 
        if b(i-3)==0 && b(i-2)==0 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=0;
            
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=1;
            
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=2;
            
        elseif b(i-3)==0 && b(i-2)==0 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=3;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=4;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=5;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=6;
            
        elseif b(i-3)==0 && b(i-2)==1 && b(i-1)==1 && b(i)==1 
            bitler1(i/4)=7;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=8;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=9;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==1 && b(i)==0  
            bitler1(i/4)=10;
            
        elseif b(i-3)==1 && b(i-2)==0 && b(i-1)==1 && b(i)==1  
            bitler1(i/4)=11;
            
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==0 && b(i)==0  
            bitler1(i/4)=12;
            
         elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==0 && b(i)==1  
            bitler1(i/4)=13;
            
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==1 && b(i)==0 
            bitler1(i/4)=14;
            
        elseif b(i-3)==1 && b(i-2)==1 && b(i-1)==1 && b(i)==1 
            bitler1(i/4)=15;
            
        end
        bits = [bits bitler1(i/4)];
end

n_bits=numel(bits); 
QAM=zeros(fs,n_bits); 

for i=1:n_bits 
t = i-1:ts:i-ts;

 if      bits(i)==0
         a=phi/4;
         qam(i)=1.4142*exp(1j*a);

 
 elseif bits(i)==1
        k=1.2490;
        qam(i)=3.1623*exp(1j*k);

 
 elseif bits(i)==2
         k=0.3218;
         qam(i)=3.1623*exp(1j*k);

 elseif  bits(i)==3
         c=phi/4;
         qam(i)=4.2426*exp(1j*c);
     
 elseif  bits(i)==4
         a= -0.7854;
         qam(i)=1.4142*exp(1j*a);
     
 elseif  bits(i)==5
         k= -1.2490;
         qam(i)=3.1623*exp(1j*k);
     
 elseif  bits(i)==6
         k= -0.3218;
         qam(i)=3.1623*exp(1j*k);
     
 elseif  bits(i)==7
         c= -0.7854;
         qam(i)=4.2426*exp(1j*c);
         
  elseif bits(i)==8
       a= 2.3562;
        qam(i)=1.4142*exp(1j*a);

 
 elseif bits(i)==9
         k=1.8925;
         qam(i)=3.1623*exp(1j*k);

 elseif  bits(i)==10
         k= 2.8198;
         qam(i)=3.1623*exp(1j*k);
     
 elseif  bits(i)==11
         c= 2.3562;
         qam(i)=4.2426*exp(1j*c);
     
 elseif  bits(i)==12
         a=-2.3562;
         qam(i)=1.4142*exp(1j*a);
     
 elseif  bits(i)==13
         k=-1.8925;
         qam(i)=3.1623*exp(1j*k);
     
 elseif  bits(i)==14
         k= -2.8198;
         qam(i)=3.1623*exp(1j*k);
         
 elseif  bits(i)==15
         c=-2.3562;
         qam(i)=4.2426*exp(1j*c);

%QAM(:,i)=qam';
 end
end
%QAM=QAM(:)';


plot(qam,'r.')
title('16-QAM Constellation Diagram')

    end
 end
end

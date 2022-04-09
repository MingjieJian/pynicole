 Module HatomicfromPe
 Implicit None
Integer, Parameter :: nlayers=4, nmaxperlayer=10, ninputs=3, noutputs=1
 Integer, Dimension(nlayers) :: Nonlin
 Integer, Dimension(0:nlayers) :: nperlayer
 Real (Kind=8), Dimension(nlayers,nmaxperlayer,nmaxperlayer) :: W
 Real (Kind=8), Dimension(nlayers,nmaxperlayer) :: Beta
 Real (Kind=8), Dimension(ninputs) :: xnorm,xmean,inputs
 Real (Kind=8), Dimension(noutputs) :: ynorm,ymean,outputs
 Real (Kind=8), Dimension(0:nlayers, nmaxperlayer) :: y
Data Nonlin(:)/1,1,1,1/
Data nperlayer(:)/3,10,10,10,1/
Data xnorm(:)/ 0.2096303283691406D+04, 0.5597151360148449D+01, 0.9996472597122192D+00/
Data xmean(:)/ 0.2687769131540399D+04,-0.3435466698725930D+01,-0.5005958698987961D+00/
Data ynorm(:)/ 0.7323348839269178D+00/
Data ymean(:)/ 0.4614444843476507D+00/
Data W(1,1,:)/-0.1171318886504880D+01,-0.3136543858714400D+00, 0.5683911376551349D-01, 0.5278299296818138D-01, 0.1464310146170390D&
&+00, 0.1069761215546140D+00,-0.5206530610222009D-01, 0.1313379920944350D+00, 0.9356030837875724D-01, 0.1052068083080850D+00/
Data W(1,2,:)/ 0.8869356273702382D-01,-0.2377869035389610D+00, 0.4653392262755320D-01, 0.1227861297147140D+00, 0.6355510856444056D&
&-01, 0.7417203870826779D-01,-0.6318995392801594D-01,-0.1423917566486950D+00, 0.1290808133531650D+00,-0.1272315213335000D+00/
Data W(1,3,:)/-0.1666562573901380D+01,-0.3757706204386950D+00, 0.1346953459520720D+00, 0.1348558818392840D+00,-0.1341941683782930D&
&+00, 0.1302193062137650D+00,-0.1290771596612950D+00, 0.4356556994248579D-01, 0.1114045377604900D+00,-0.1198257488487690D+00/
Data W(1,4,:)/-0.2272710820620170D+00, 0.1003758191593290D+02, 0.5140665657566800D+01,-0.1043982784871880D+00, 0.1564515723337460D&
&+00, 0.8022460709458774D-01,-0.8918452553920135D-01, 0.5037273437419098D-01, 0.1233869952610146D-01,-0.6304515930608312D-02/
Data W(1,5,:)/-0.1127208901275460D+00,-0.4644551266370984D-01, 0.4541865449467419D-02, 0.2712307505850869D-01, 0.8445428835016636D&
&-01,-0.1745044402293731D-01, 0.6625132038103604D-01, 0.9313732978336656D-01,-0.1938433288081942D-01,-0.9210029357869073D-01/
Data W(1,6,:)/ 0.1303121551827340D+00,-0.1911409491160827D-01,-0.2003151129507066D-01,-0.9124088788028764D-01,-0.6063618735293769D&
&-01,-0.7494948534785498D-01, 0.7978297408440610D-01, 0.1574642027101912D-01, 0.4323590656677449D-01, 0.4142911264404125D-01/
Data W(1,7,:)/ 0.1542717916654810D+00, 0.7916381167502720D+00,-0.9436863930281043D-01, 0.1103530562052453D-01,-0.9671601980590130D&
&-01, 0.1479646606148060D+00, 0.2821706774275322D-01,-0.1525408979016770D+00,-0.5870997946481357D-01, 0.1685589961979228D-01/
Data W(1,8,:)/ 0.4167530280815130D+00,-0.1035153385296770D+02,-0.5431264401738030D+01,-0.1237081555421269D-01, 0.1067110686805580D&
&+00,-0.8571256817315648D-01, 0.3905902772629295D-01,-0.1437392147084770D+00, 0.7343594709842007D-02, 0.9793066619565594D-01/
Data W(1,9,:)/-0.1254636343404320D+00, 0.9957069155655489D+01, 0.5077534116977540D+01, 0.7780236164642972D-01,-0.7618807717297932D&
&-01, 0.5416979948700212D-01, 0.4900719015678941D-01, 0.1376998072406990D+00, 0.1490247746281990D+00, 0.3011298416621610D-03/
Data W(1,10,:)/-0.1567345471295230D+01,-0.8339556649350340D+00, 0.2788769353528280D+00,-0.1047635627080420D+00, 0.1348571491160950&
&D+00, 0.1124677032662943D-01, 0.1195053452376447D-01, 0.2051393875294429D-01,-0.4119920630666821D-01,-0.3765835044604780D-02/
Data W(2,1,:)/-0.3668518764409280D+01, 0.4263999624641120D+01, 0.2105325146145160D+01,-0.5093165306569630D+00, 0.6185534661927070D&
&+01, 0.3478769403900520D+01,-0.7848202875023290D+00,-0.2606777761303400D+00, 0.2581448032457460D+00,-0.5968073244428220D+00/
Data W(2,2,:)/ 0.7333605348146850D+00, 0.5642783748293281D-01,-0.2859792715542250D+00,-0.5305771261273711D-01, 0.9850551047149891D&
&-02,-0.1155030750460830D+00, 0.1654334271781330D+00,-0.3710263377822396D-01,-0.4533949752675719D-01,-0.1639090347915752D-01/
Data W(2,3,:)/ 0.3139059191307910D+00, 0.1488517134090810D+00,-0.6918266627364093D-01, 0.3652239701640740D-01, 0.8504015529864000D&
&-02,-0.1039178584824020D+00,-0.9656563220287892D-01, 0.5455728994522440D-01,-0.7376318126619615D-01,-0.4413761242156941D-01/
Data W(2,4,:)/ 0.5167292415970520D+00,-0.4861328963964922D-01,-0.2141622667123293D-01, 0.4116462645530085D-01,-0.2579837170229010D&
&+00,-0.1250689558459250D-02, 0.5847600534306396D-01, 0.2123292625375770D+00,-0.3015700664674373D-01, 0.2036842861564720D+00/
Data W(2,5,:)/ 0.1217987969082730D+01,-0.4796659297681550D+00, 0.1155005755011470D+01, 0.2685177100472430D+00, 0.4211754954357450D&
&+00,-0.2379155962731030D+00,-0.9708117778788622D-01, 0.1496840617506420D+00,-0.1203394058285850D+00,-0.8236548589715500D+00/
Data W(2,6,:)/ 0.1249003433925060D+00,-0.2176245329529205D-01, 0.5262252608987966D-01, 0.6905608836578464D-02, 0.1359194795554010D&
&+00, 0.1360823127854090D+00, 0.1179098006613620D+00,-0.1314081991785400D+00,-0.9966105304373959D-01, 0.3527698300111892D-01/
Data W(2,7,:)/-0.2457340465632291D-01,-0.9559345502417449D-01, 0.1212994094665431D-01,-0.1089986946911590D+00,-0.8808069390949660D&
&-01,-0.1431123239733330D+00,-0.1385340946396200D+00, 0.1149571562724970D+00,-0.7279132262314298D-02, 0.1900529536929900D+00/
Data W(2,8,:)/-0.1042361105429470D+01,-0.2302957392444620D+00, 0.4826540838200840D+00,-0.1732339474823420D+00, 0.1164634030908850D&
&+00, 0.4524012045091019D-01,-0.3280476775537080D+00,-0.2061536143094444D-01, 0.6287478548283870D-01,-0.1296397675659290D+00/
Data W(2,9,:)/ 0.8688767326576100D+00, 0.3071898347523410D+00,-0.4043193624556370D+00,-0.7167784570491764D-01,-0.2075861255749070D&
&+00, 0.1373584067029930D+00, 0.2132254251235110D+00,-0.4530604160459848D-01, 0.1079526504358610D+00, 0.6726996932413198D-01/
Data W(2,10,:)/-0.9746629308801910D+00,-0.2612995288086120D+00, 0.5944456902507110D+00, 0.4222592834563451D-01, 0.3002037627721800&
&D+00,-0.1030900034118780D+00,-0.1290240722101830D+00, 0.1136760989306400D+00, 0.1072476084197850D+00,-0.7387046668366969D-01/
Data W(3,1,:)/ 0.7423962045351900D+01,-0.3225908626037980D+01,-0.1404591486335720D+01,-0.1020093652298780D+01, 0.2709919917715460D&
&+01,-0.8878819783515980D+00,-0.3288137887794160D+00, 0.4188523715417840D+01,-0.5112214039722820D+01, 0.3971208623671820D+01/
Data W(3,2,:)/-0.9183009149065290D-01,-0.1163893164648400D+00,-0.7608013938130290D-01, 0.5915024285593593D-01,-0.8200488421457246D&
&-01, 0.1103290441369770D+00, 0.5543013292867688D-01,-0.1080027148741400D+00,-0.4802218165848392D-01, 0.3610738423770943D-01/
Data W(3,3,:)/-0.2629365961456552D-01, 0.9335674743927473D-01,-0.3117797430916614D-01, 0.1410796830160981D-01,-0.4982048559230980D&
&-01, 0.8471333380791063D-01,-0.9397039597408043D-01, 0.8265731692713091D-01, 0.1121232649096272D-01,-0.1447397907879551D-01/
Data W(3,4,:)/-0.9519410267148451D-01, 0.7986919823798827D-01,-0.6548846538989031D-01,-0.9410755608522087D-01, 0.4239556126528737D&
&-01,-0.5382995401480393D-01, 0.8371001974198586D-01,-0.2312654681926423D-01,-0.1350940372563320D+00, 0.5807214355462019D-01/
Data W(3,5,:)/ 0.1488615103229110D+00, 0.9868521791727634D-01,-0.1077638080146610D+00,-0.9476481611226024D-01,-0.1821521053166250D&
&-01, 0.1239882204041110D+00, 0.9250913981408743D-01, 0.1371422242925150D+00,-0.1450183863392240D+00,-0.8497621246141039D-01/
Data W(3,6,:)/-0.1166152119432653D-01,-0.6293737620581569D-01,-0.2810328849118730D-01, 0.2195437467112550D-01, 0.1201648795450390D&
&+00, 0.1224200270503890D+00, 0.6360218835445883D-01,-0.6415532773477875D-01, 0.4875455463698584D-02, 0.2207692502145673D-01/
Data W(3,7,:)/-0.1139665415612154D-01, 0.4261613398967163D-01, 0.2818471610659038D-01, 0.4551086237735123D-01,-0.4741724502592042D&
&-01, 0.1797346798201891D-01, 0.8456782036053312D-01, 0.5201700144483958D-01,-0.1569161092444520D+00,-0.2302337213887385D-01/
Data W(3,8,:)/ 0.1476330101796840D+00,-0.1368032190575330D+00, 0.7445714904306601D-01,-0.5130335452069901D-01,-0.1370401937837290D&
&+00, 0.1040514242644550D+00,-0.2712056701079326D-01,-0.1408072380130340D+00, 0.1274941424379460D+00,-0.6225292612601786D-01/
Data W(3,9,:)/-0.1205253660496750D+00,-0.3140318365276343D-01, 0.1417668442819320D+00,-0.2903640137088739D-01, 0.3103236264268124D&
&-01, 0.7554603050512732D-01, 0.1511261219092090D+00,-0.8557774396859891D-01,-0.4299295799841479D-01,-0.6078893278947863D-01/
Data W(3,10,:)/ 0.1469237672118040D+00,-0.8291887374583289D-02,-0.2814839771435870D-01,-0.9156859894877610D-01, 0.1249761314463650&
&D+00,-0.1387531107088330D+00, 0.1089234076753660D+00,-0.5586950410155769D-01,-0.4770298340343288D-01,-0.7276103097208692D-01/
Data W(4,1,:)/ 0.3452652767560930D+00, 0.1101454944022380D+00, 0.2356135157753070D+00, 0.1208707925568069D-01,-0.6332986058056354D&
&-01, 0.1755920848518920D+00, 0.1260892694346080D+00,-0.5629771996752451D-02,-0.4260370830468241D-01, 0.4847062981171281D-02/
Data W(4,2,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,3,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,4,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,5,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,6,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,7,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,8,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,9,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D&
&+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data W(4,10,:)/ 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000&
&D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00, 0.0000000000000000D+00/
Data Beta(1,:)/ -0.428239286511180, -0.114452173301002, -1.710521259847632E-002, -5.066570577083361E-002, -0.157517918903710, -5.5&
&48990844718387E-002, -2.743106410699187E-002,  0.152438139662383, -2.303593608004781E-002,  0.227094087297994/
Data Beta(2,:)/ -0.289188091856287, -9.240874197771346E-003, -5.926017660833833E-003, -1.492478636501965E-002,  3.199805865042242E&
&-002,  2.624341437275709E-002,  1.657433785569427E-002,  3.788849587540542E-002,  3.428870201695522E-002,  4.912378106517146E-002/
Data Beta(3,:)/  1.260741315211286E-002,  3.503483540801113E-002,  4.033895096259973E-002,  3.463544701335341E-002, -2.93375811923&
&7471E-002,  1.420348804062595E-002, -3.630032534577794E-002, -3.073028374716790E-002,  4.667018144876264E-002, -1.393652441006782E&
&-002/
Data Beta(4,:)/ -5.558593028885016E-002,  0.000000000000000E+000,  0.000000000000000E+000,  0.000000000000000E+000,  0.00000000000&
&0000E+000,  0.000000000000000E+000,  0.000000000000000E+000,  0.000000000000000E+000,  0.000000000000000E+000,  0.000000000000000E&
&+000/
 End Module HatomicfromPe
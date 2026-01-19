(* ::Package:: *)

(* ::Title:: *)
(*ML Amplitudes: Data Generation*)


(* ::Text:: *)
(*In this notebook we generate data to train a neural network*)


Quit[]


(* ::Section::Closed:: *)
(*Preliminaries*)


(* ::Text:: *)
(*Import massive spinor helicity (NB: select directory of 'MSHfinal.m' file)*)


<<"/Users/paolo/Desktop/Packages/MSHfinal.m"


(* ::Section:: *)
(*Dot products to field strengths*)


(* ::Text:: *)
(*NEED: Preliminaries*)


(* ::Text:: *)
(*IDEA:*)
(*1. define and generate gauge-invariant structures in terms of p.p and p.F.(...).F.p terms*)
(*2. rewrite them in terms of ordinary dot products*)
(*3. define a set of scrambling rules to complicate the dot product expression*)
(*4. collect each pair (simple,scrambled) in a text file as training/testing data for a neural net*)


(* ::Subsection::Closed:: *)
(*Useful functions*)


(* ::Text:: *)
(*Redefine weight counting functions*)


ClearAll[covweights,massdim];
covweights[expr_,i_]:=Exponent[expr/.{pol[i]->\[Epsilon] pol[i]},\[Epsilon]];
massdim[expr_,masses_List]:=Exponent[expr/.Join[{vec[x_]:>\[Epsilon] vec[x]},((#->\[Epsilon] #)&/@DeleteCases[Union[masses],0])],\[Epsilon]];


(* ::Text:: *)
(*Momentum conservation replacement*)


ClearAll[pcons];
pcons[N_,vec[x_]]:=-(Sum[vec[i],{i,N}]/.{vec[x]->0});


(* ::Subsection::Closed:: *)
(*Scrambling*)


(* ::Text:: *)
(*NEED: Useful functions*)


(* ::Text:: *)
(*Kinematics choice*)


kinrule = {dot[vec[x_]]:>m[x]^2,dot[pol[x_]]:>0,dot[pol[x_],vec[x_]]:>0};


(* ::Text:: *)
(*List of all possible dot products*)


ClearAll[EEterms,EPterms,PPterms,zeroterms];
(*create list of all allowed dot products and save it each time it is called to avoid computing again*)
EEterms[N_]:=(EEterms[N]=DeleteCases[Flatten@Table[dot[pol[i],pol[j]],{i,1,N},{j,1,i-1}]//.kinrule,0]);
EPterms[N_]:=(EPterms[N]=DeleteCases[Flatten@Table[dot[pol[i],vec[j]],{i,1,N},{j,1,N}]//.kinrule,0]);
PPterms[N_]:=(PPterms[N]=DeleteCases[Flatten@Table[dot[vec[i],vec[j]],{i,1,N},{j,1,i-1}]//.kinrule,0]);
(*create list of zeros*)
zeroterms[N_,weights_List,masses_List]/;(Length@masses==N&&Length@weights==N):=(zeroterms[N]=Join[Table[dot[vec[i],-(Sum[vec[j],{j,1,N}]/.vec[i]->0)]-m[i]^2,{i,N}],Table[If[weights[[i]]==0,Nothing,dot[pol[i],pcons[N,vec[i]]]],{i,N}]]/.Thread[Table[m[i],{i,N}]->masses]);


(* ::Text:: *)
(*Create random monomial V1:*)
(*- generate all allowed monomials and cache them*)
(*- pick one at random from the cached list*)


ClearAll[allMono,randomMono];
$valuesMono = {};(*save values called, for debugging*)
(*create list of all allowed monomials and save it each time it is called to avoid computing again*)
allMono[N_,weights_List,masses_List,mdim_]/;(Length@masses==N&&Length@weights==N):=(AppendTo[$valuesMono,{N,weights,masses,mdim}];allMono[N,weights,masses,mdim]=DeleteCases[AnsatzCovFull[Join[EEterms[N],EPterms[N],PPterms[N]],weights,mdim,masses],0]);
(*pick a random choice from the above list*)
randomMono[N_,weights_List,masses_List,mdim_]:=(RandomChoice@allMono[N,weights,masses,mdim]);


(* ::Text:: *)
(*Create random monomial V2:*)
(*- generate a single random monomial, to avoid generating huge lists*)


(* ::Text:: *)
(*Scramble a given polynomial or monomial*)
(*NB: it works also for negative powers, i.e. mono --> single rational function, poly --> sum of rational functions*)
(*NB: input must be given in expanded form (I think?...) ie after calling Expand or ExpandAll*)


ClearAll[scramble1,scramble2,scramble3,scramble]
(*multiply by 1*)
scramble1[poly_Plus,N_,masses_List,maxD_:0]:=MapAt[scramble1[#,N,masses,maxD]&,poly,RandomChoice@Table[i,{i,1,Length@poly}]];
scramble1[mono_Times,N_,masses_List,maxD_:0]:=With[{pmono=randomMono[N,Table[0,N],masses,2RandomChoice@Join[{0},Range[maxD]]]},
	Times[((decomp[pmono]/.{dot[vec1_,vec2_]:>RandomChoice[{dot[pcons[N,vec1],vec2],dot[vec1,pcons[N,vec2]]}]})/.List->Times)/pmono,mono]//.kinrule/.Thread[Table[m[i],{i,N}]->masses]
];
(*add zero*)
scramble2[poly_Plus,N_,weights_List,masses_List,maxD_:0]:=MapAt[scramble2[#,N,weights,masses,maxD]&,poly,RandomChoice@Table[i,{i,1,Length@poly}]];
(*OLD: scramble2[mono_Times,N_,weights_List,masses_List,maxD_:0]:=With[{mdim=massdim[mono,Union@masses],denom=RandomChoice@Join[{0},Range[maxD]]},
	(mono+(Times[#,randomMono[N,weights-Table[covweights[#,i],{i,N}],masses,mdim+2denom-massdim[#,Union@masses]]]&@(RandomChoice@zeroterms[N,weights,masses]))/randomMono[N,Table[0,N],masses,2denom])//.kinrule/.Thread[Table[m[i],{i,N}]->masses]
];*)
scramble2[ratio_Times,N_,weights_List,masses_List,maxD_:0]:=With[{mono = Numerator@ratio, deno = Denominator@ratio},
	(mono+(Times[#,randomMono[N,weights-Table[covweights[#,i],{i,N}],masses,massdim[mono,Union@masses]+2RandomChoice@Join[{0},Range[maxD]]-massdim[#,Union@masses]]]&@(RandomChoice@zeroterms[N,weights,masses]))/randomMono[N,Table[0,N],masses,2RandomChoice@Join[{0},Range[maxD]]])/deno//.kinrule/.Thread[Table[m[i],{i,N}]->masses]
];
(*apply momentum conservation*)
scramble3[poly_Plus,N_,masses_List]:=MapAt[scramble3[#,N,masses]&,poly,RandomChoice@Table[i,{i,1,Length@poly}]];
scramble3[mono_Times,N_,masses_List]:=Quiet@With[{p=RandomChoice[Union@Cases[mono,_vec,\[Infinity]]]},
	ReplacePart[mono,RandomChoice[Position[mono, p]] -> (-Sum[vec[j],{j,N}]/.p->0)]//.kinrule/.Thread[Table[m[i],{i,N}]->masses]
];
(*put together*)
scramble[type_,mono_,N_,weights_List,masses_List,maxD_:0]/;MemberQ[{1,2,3},type]:=Switch[type,
1,ExpandAll@scramble1[mono,N,masses,maxD],
2,ExpandAll@scramble2[mono,N,weights,masses,maxD],
3,ExpandAll@scramble3[mono,N,masses]
];


(* ::Subsection::Closed:: *)
(*Gauge-invariant structures*)


(* ::Text:: *)
(*NEED: Useful functions, Scrambling*)


(* ::Text:: *)
(*Here we create:*)
(*- numerators in terms of field strengths F_i*)
(*- rational functions using such numerators and momentum dot products as denominators*)


(* ::Text:: *)
(*Note: the most general structures are of form { V1 . F[i_ 1] . (...) . F[i_n].V2 , tr(F[i_1].F[i_2]) } for arbitrary particles i_k and vectors V1,V2*)


Fkillrule = {
trF[x_,F[y_],x_]:>Nothing,
trF[vec[x_],F[x_],right___]:>Nothing,trF[left___,F[x_],vec[x_]]:>Nothing,
trF[left___,F[x_],F[x_],right___]:>Nothing,
trF[x_,seq:F[__]..,x_] /; (lst={seq}/.F[i_]:>i; lst===Reverse[lst] && OddQ[Length@lst]) :> Nothing,
trF[x_,seq:F[__]..,y_] /; (lst={x,seq,y}; !OrderedQ[{lst,Reverse[lst]}]) :> Nothing,
trFF[left___,F[x_],F[x_],right___]:>Nothing,trFF[F[x_],middle___,F[x_]]:>Nothing,
trFF[seq:F[__]..] /; (lst={seq}/.F[i_]:>i; lst===Reverse[lst] && OddQ[Length@lst]) :> Nothing,
trFF[seq:F[__]..] /; (lst={seq}; !OrderedQ[{lst,Reverse[lst]}]) :> Nothing
};
ClearAll[basisF];
basisF[weights_List] := basisF[weights] = Module[{vFv = {},FF = {},N=Length@weights},
	vFv = DeleteDuplicates[Flatten[Table[trF[vec[a],Sequence@@Table[F[x[i]],{i,n}],vec[b]],{n,Plus@@weights}]/.(Thread[Table[x[i],{i,Plus@@weights}]->#]&/@Permutations@Flatten@Table[Table[i,weights[[i]]],{i,Length@weights}])]/.{trF[left___,F[x_],F[x_],right___]:>Nothing}];
	vFv = Flatten[vFv/.Flatten[Table[{a->i,b->j},{i,1,N},{j,1,N}],1]]//.Fkillrule;
	(*FF = Flatten@Table[trFF[F[i],F[j]],{j,1,N},{i,1,j-1}];*)
	FF = DeleteDuplicates[Flatten[Table[trFF[Sequence@@Table[F[x[i]],{i,n}]],{n,Plus@@weights}]/.(Thread[Table[x[i],{i,Plus@@weights}]->#]&/@Permutations@Flatten@Table[Table[i,weights[[i]]],{i,Length@weights}])]/.{trFF[left___,F[x_],F[x_],right___]:>Nothing,trFF[F[x_],middle___,F[x_]]:>Nothing}];
	FF = Flatten[FF/.Flatten[Table[{a->i,b->j},{i,1,N},{j,1,N}],1]]//.Fkillrule;
	Union[vFv,FF]
];


(* ::Text:: *)
(*Now create a list of random rationals, where the numerator is a monomial in terms of trF and trFF structures (with the appropriate weights + dimensions) and the denominator is just p.p products*)
(*(CREATED BY CHATGPT o4-mini-high)*)


(* helper: compute the weight\[Hyphen]vector of a single basis element *)
ClearAll[weightVector];
weightVector[expr_, N_Integer] := Module[{fs = Cases[expr, F[i_] :> i, \[Infinity]], v = ConstantArray[0, N]},
  Scan[(v[[#]]++)&,fs];
  v
];

(* enumerate every product of basis elements whose total F-counts = weights *)
ClearAll[allMonomialsF];
allMonomialsF[weights_List] := allMonomialsF[weights] = Module[{N = Length[weights], B, wvList, rec},
  B = basisF[weights];
  wvList = Map[{#, weightVector[#, N]} &, B];
  rec[ws_] := If[
  Total[ws] == 0,{{}},(* empty product \[DoubleRightArrow] the \[OpenCurlyDoubleQuote]1\[CloseCurlyDoubleQuote] monomial *)
    Flatten[Table[Module[{b = wvList[[i, 1]], wB = wvList[[i, 2]]},If[And @@ Thread[wB <= ws],(* use b once, then fill out the rest recursively *)Prepend[#, b] & /@ rec[ws - wB],{}]],{i, Length[B]}],1]
  ];
  Times@@@(rec[weights])
];
(* pick a random monomial (as a Times[\[Ellipsis]] of basis elements) *)
ClearAll[randomMonomialF];
randomMonomialF[weights_List] := Module[
  {all = allMonomialsF[weights]},
  If[all === {}, (* no way to build anything except the identity *)1,RandomChoice[all]]
];
(*construct a random rational function*)
ClearAll[randomRatF,mdimF];
mdimF[expr_]:=Total@Cases[expr,(vec|F)[_]^n_.:>n,Infinity];
randomRatF[weights_List, masses_List, dendim_Integer,numlength_Integer,maxint_Integer] := Module[{N = Length@weights, numdim = 4-Length@weights+dendim, denom, Fmonos, nummonos},
denom = randomMono[N,Table[0,N],masses,dendim](*denominator w mass dimension dendim*);
Fmonos = Select[allMonomialsF[weights],mdimF@# <= numdim &](*all F monomials with mass dimension <= numdim*);
nummonos = Table[Times[#,randomMono[N,Table[0,testN],masses,numdim - mdimF@#]]&@RandomChoice[Fmonos],numlength](*numlength-many F monomials times random p.p term to make mass dimension of each = numdim*);
Plus@@(Times[#,RandomInteger[{1,maxint}]]&/@nummonos)/denom(*sum of numerator monomials, divided by denominator*)
];


(* ::Text:: *)
(*Finally create a way to convert F structures into ordinary dot products*)


FtoDot = {
trF[x1_,seq:F[__]..,x2_] :>  Expand[(lst={seq};len=Length@lst;dot[x1,ind[1]]dot[x2,ind[len+1]]Product[lst[[j]][ind[j],ind[j+1]],{j,1,len}])/.Frule],
trFF[F[x_],F[y_]] :> Expand[F[x][ind[1],ind[2]]F[y][ind[2],ind[1]]/.Frule],
trFF[seq:F[__]..] :>  Expand[(lst={seq};len=Length@lst;Product[lst[[j]][ind[j],ind[j+1]],{j,1,len}]/.ind[len+1]->ind[1])/.Frule]
};


(* ::Subsection:: *)
(*Data generation*)


(* ::Text:: *)
(*NEED: Useful functions, Scrambling, Testing, Gauge-invariant structures*)


(* ::Text:: *)
(*Create data*)


(*five point scrambled*)
testN = 5; testweights = {0,0,1,1,1}; testmasses = {M,M,0,0,0}; testdendim = 8; testmaxD = 2;
datalist = {};
Monitor[Do[
target = randomRatF[testweights,testmasses,testdendim,RandomInteger[{1,5}],10];
exptarget = Expand[target/.FtoDot];
scram1 = scramble[sc1=RandomChoice[{1,2,3}],exptarget,testN,testweights,testmasses,testmaxD];
scram2 = scramble[sc2=RandomChoice[{1,2,3}],scram1,testN,testweights,testmasses,testmaxD];
AppendTo[datalist,{target,{sc1,scram1},{sc2,scram2}}];,
{ii,50000}],ii]//AbsoluteTiming


(* ::Subsection::Closed:: *)
(*Save Data*)


(* ::Text:: *)
(*NEED: Useful functions, Scrambling, Testing, Gauge-invariant structures, Data generation*)


(* ::Text:: *)
(*Save to file*)


datalist >> "/Users/paolo/Desktop/Projects/AmplitudesFromML/Data/cluster.wl";


(* ::Text:: *)
(*Save to file as string*)


expr = datalist/.{trF[x___]:>CenterDot[x],trFF[x___]:>Tr@CenterDot[x]}/.dot[x_,y_]:>x\[CenterDot]y/.vec[x_]:>Subscript[p, x]/.F[x_]:>Subscript[F, x]/.pol[x_]:>Subscript[e, x];
exprUnderscore=expr/. Subscript[a_,b_]:>ToString[a]<>"_"<>ToString[b];
exprStringList = StringReplace[ToString[#,InputForm],{"\""->"","{"->"","}"->"\n","["->"(","]"->")"}]&/@exprUnderscore;
exprString=StringJoin@exprStringList;
Export["/Users/paolo/Desktop/Projects/AmplitudesFromML/Data/cluster.csv",exprString,"Text"];

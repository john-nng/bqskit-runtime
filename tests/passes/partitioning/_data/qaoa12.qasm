OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
rz(pi*3.9021505871) q[1];
rz(pi*3.9021505871) q[3];
rz(pi*-3.9021505871) q[5];
rz(pi*-3.9021505871) q[7];
rz(pi*-3.9021505871) q[9];
rz(pi*3.9021505871) q[11];
cx q[1],q[0];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[6];
cx q[9],q[8];
cx q[11],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rz(pi*-3.9021505871) q[2];
rz(pi*-3.9021505871) q[4];
rz(pi*3.9021505871) q[6];
rz(pi*-3.9021505871) q[8];
rz(pi*3.9021505871) q[10];
cx q[2],q[1];
cx q[4],q[3];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[9];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
rz(pi*-3.9021505871) q[1];
rz(pi*3.9021505871) q[3];
rz(pi*3.9021505871) q[5];
rz(pi*-3.9021505871) q[7];
rz(pi*-3.9021505871) q[9];
rz(pi*-3.9021505871) q[11];
cx q[1],q[0];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[6];
cx q[9],q[8];
cx q[11],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rz(pi*-3.9021505871) q[2];
rz(pi*-3.9021505871) q[4];
rz(pi*-3.9021505871) q[6];
rz(pi*-3.9021505871) q[8];
rz(pi*-3.9021505871) q[10];
cx q[2],q[1];
cx q[4],q[3];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[9];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
rz(pi*-3.9021505871) q[1];
rz(pi*3.9021505871) q[3];
rz(pi*3.9021505871) q[5];
rz(pi*-3.9021505871) q[7];
rz(pi*-3.9021505871) q[9];
rz(pi*-3.9021505871) q[11];
cx q[1],q[0];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[6];
cx q[9],q[8];
cx q[11],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rz(pi*-3.9021505871) q[2];
rz(pi*3.9021505871) q[4];
rz(pi*-3.9021505871) q[6];
rz(pi*-3.9021505871) q[8];
rz(pi*-3.9021505871) q[10];
cx q[2],q[1];
cx q[4],q[3];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[9];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
rz(pi*-3.9021505871) q[1];
rz(pi*3.9021505871) q[3];
rz(pi*3.9021505871) q[5];
rz(pi*3.9021505871) q[7];
rz(pi*-3.9021505871) q[9];
rz(pi*3.9021505871) q[11];
cx q[1],q[0];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[6];
cx q[9],q[8];
cx q[11],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rz(pi*3.9021505871) q[2];
rz(pi*-3.9021505871) q[4];
rz(pi*-3.9021505871) q[6];
rz(pi*-3.9021505871) q[8];
rz(pi*3.9021505871) q[10];
cx q[2],q[1];
cx q[4],q[3];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[9];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
rz(pi*3.9021505871) q[1];
rz(pi*3.9021505871) q[3];
rz(pi*-3.9021505871) q[5];
rz(pi*3.9021505871) q[7];
rz(pi*3.9021505871) q[9];
rz(pi*-3.9021505871) q[11];
cx q[1],q[0];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[6];
cx q[9],q[8];
cx q[11],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rz(pi*-3.9021505871) q[2];
rz(pi*-3.9021505871) q[4];
rz(pi*-3.9021505871) q[6];
rz(pi*-3.9021505871) q[8];
rz(pi*-3.9021505871) q[10];
cx q[2],q[1];
cx q[4],q[3];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[9];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
rz(pi*3.9021505871) q[1];
rz(pi*3.9021505871) q[3];
rz(pi*3.9021505871) q[5];
rz(pi*3.9021505871) q[7];
rz(pi*-3.9021505871) q[9];
rz(pi*-3.9021505871) q[11];
cx q[1],q[0];
cx q[3],q[2];
cx q[5],q[4];
cx q[7],q[6];
cx q[9],q[8];
cx q[11],q[10];
cx q[0],q[1];
cx q[2],q[3];
cx q[4],q[5];
cx q[6],q[7];
cx q[8],q[9];
cx q[10],q[11];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rx(pi*2.2474367096) q[0];
rx(pi*2.2474367096) q[11];
rz(pi*3.9021505871) q[2];
rz(pi*3.9021505871) q[4];
rz(pi*3.9021505871) q[6];
rz(pi*-3.9021505871) q[8];
rz(pi*-3.9021505871) q[10];
cx q[2],q[1];
cx q[4],q[3];
cx q[6],q[5];
cx q[8],q[7];
cx q[10],q[9];
cx q[1],q[2];
cx q[3],q[4];
cx q[5],q[6];
cx q[7],q[8];
cx q[9],q[10];
rx(pi*2.2474367096) q[1];
rx(pi*2.2474367096) q[2];
rx(pi*2.2474367096) q[3];
rx(pi*2.2474367096) q[4];
rx(pi*2.2474367096) q[5];
rx(pi*2.2474367096) q[6];
rx(pi*2.2474367096) q[7];
rx(pi*2.2474367096) q[8];
rx(pi*2.2474367096) q[9];
rx(pi*2.2474367096) q[10];

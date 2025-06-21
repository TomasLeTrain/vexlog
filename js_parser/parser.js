// https://stackoverflow.com/questions/71080938/decode-a-prepended-varint-from-a-byte-stream-of-unknown-size-in-javascript-nodej
function readVarUInt(buffer) {
    let value = 0;
    let length = 0;
    let currentByte;

    while (true) {
        currentByte = buffer[length];
        value |= (currentByte & 0x7F) << (length * 7);
        length += 1;
        if (length > 5) {
            throw new Error('VarInt exceeds allowed bounds.');
        }
        if ((currentByte & 0x80) != 0x80) break;
    }
    return value;
}

function readVarInt(buffer) {
    let value = readVarUInt(buffer);
    let res = value >> 1;
    return (value & 1) ? ~res : res;
}


// https://stackoverflow.com/questions/5678432/decompressing-half-precision-floats-in-javascript#8796597
function decodeFloat16 (binary) {"use strict";
    var exponent = (binary & 0x7C00) >> 10,
        fraction = binary & 0x03FF;
    return (binary >> 15 ? -1 : 1) * (
        exponent ?
        (
            exponent === 0x1F ?
            fraction ? NaN : Infinity :
            Math.pow(2, exponent - 15) * (1 + fraction / 0x400)
        ) :
        6.103515625e-5 * (fraction / 0x400)
    );
};

function readMagic(){
    buffer[0]
}

function generationReader(buffer){
    readMagic(buffer);
}

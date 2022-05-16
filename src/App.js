import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import {useCallback, useEffect, useRef, useState} from "react";
// import Camera, { FACING_MODES } from 'react-html5-camera-photo';
import 'react-html5-camera-photo/build/css/index.css';

// import src from './t2.jpg';

// https://github.com/tensorflow/tfjs/issues/6026
tflite.setWasmPath('tflite_wasm/');

const MODEL = './model.tflite';
const COLOR = {
    0: '#ff0000',
    1: '#ff5900',
    2: '#ff8800',
    3: '#ffc400',
    4: '#a2ff00',
    5: '#00ff66',
    6: '#00ffbb',
    7: '#008cff',
    8: '#1100ff',
    9: '#9900ff',
    10: '#ff1414',
    11: '#ff0073',
    12: '#69ff37',
    13: '#080c57',
    14: '#fff600',
    15: '#ff2e2e',
    16: '#70f2ff',
};

const SHIFT = {
    0: {
        height: 0.3,
        width: -0.1
    },
    1: {
        height: 0.2,
    },
    2: {
        height: 0.2,
    },
    3: {
        height: 0.2,
        width: -0.1
    },
    8: {
        height: 0.3,
        width: -0.2
    },
    10: {
        height: 0.2,
    },
    13: {
        height: 0.2,
    },
    14: {
        width: 0.2,
    },
    16: {
        height: 0.4,
    }
}

const K = 0.2;
const MODEL_SIZE = 320;


function App() {
    const videoRef = useRef(null);
    // const canvasRef = useRef(null);
    // const imageRef = useRef(null);


    const imageRef = useRef(null);
    const wrapRef = useRef(null);
    const [model, setModel] = useState(null);
    // eslint-disable-next-line no-unused-vars
    const [photo, setPhoto] = useState(null);
    const [isLoadPhoto, setIsLoadPhoto] = useState(false);
    // eslint-disable-next-line no-unused-vars
    const [box, setBox] = useState({});
    // eslint-disable-next-line no-unused-vars
    const [points, setPoints] = useState({});
    const [size, setSize] = useState({width: undefined, height: 0});

    useEffect(() => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices
                .getUserMedia({
                    audio: false,
                    video: {
                        facingMode: "environment"
                    }
                })
                .then(async (stream) => {
                    window.stream = stream;
                    videoRef.current.srcObject = stream;

                    const clientWidth = document.getElementsByTagName('body')[0].clientWidth - 6;
                    let {width, height} = stream.getTracks()[0].getSettings();

                    setSize({
                        width: clientWidth,
                        height: clientWidth * 1.2,
                    });
                    console.log(`${width}x${height}`); // 640x480

                    const devices = await navigator.mediaDevices.enumerateDevices();




                    console.log(devices)
                    return new Promise((resolve) => {
                        videoRef.current.onloadedmetadata = () => {
                            resolve();
                        };
                    });
                });
        }
    }, []);

    useEffect(() => {
        if (model) {
            const detectFrame = () => {
                tf.engine().startScope();

                const img = tf.browser.fromPixels(videoRef.current);
                const smallImg = tf.image.resizeBilinear(img, [MODEL_SIZE, MODEL_SIZE]);
                const input = tf.sub(tf.div(tf.expandDims(smallImg), 127.5), 1);

                console.log('W ', videoRef.current.videoWidth);
                console.log('H', videoRef.current.videoHeight);
                let outputTensor = model.predict(input);
                // console.timeEnd('predict');

                // console.log('size',size)
                const boxes = outputTensor['StatefulPartitionedCall:3'].arraySync()[0];
                const classes = outputTensor['StatefulPartitionedCall:2'].arraySync()[0];
                const probability = outputTensor['StatefulPartitionedCall:1'].arraySync()[0];


                let result = {};
                let resultPoints = {};

                for (let index in probability) {
                    if (probability[index] < K) {
                        break;
                    }
                    const classKey = classes[index];

                    if (result[classKey] === undefined) {
                        const [y1, x1, y2, x2] = boxes[index]
                        result[classKey] = {
                            top: y1 * size.height,
                            left: x1 * size.width,
                            right: size.width - (x2 * size.width),
                            bottom: size.height - (y2 * size.height),
                            probability: probability[index].toFixed(2),
                            color: COLOR[classKey]
                        };

                        const leftB = x1 * size.width;
                        const rightB = x2 * size.width;
                        let widthB = rightB - leftB;
                        let centerWidthB = widthB / 2;

                        if (SHIFT[classKey]?.width) {
                            centerWidthB+= widthB * SHIFT[classKey]?.width
                        }
                        const left = leftB + centerWidthB;

                        const topB = y1 * size.height;
                        const bottomB = y2 * size.height;
                        let heightB = bottomB - topB;
                        let centerHeightB = heightB / 2;

                        if (SHIFT[classKey]?.height) {
                            centerHeightB+= heightB * SHIFT[classKey]?.height
                        }
                        const top = topB + centerHeightB;

                        resultPoints[classKey] = {
                            top,
                            left,
                            probability: probability[index].toFixed(2),
                            color: COLOR[classKey]
                        }
                    }
                }
                setBox(result);
                setPoints(resultPoints);

                requestAnimationFrame(() => {
                    detectFrame();
                });
                tf.engine().endScope();

            }

            detectFrame();
        }
    }, [size, model]);

    useEffect(() => {
        tflite.loadTFLiteModel(MODEL).then(_model => {
            setModel(_model);
        })
    }, [setModel, setSize]);

    const predict = useCallback(() => {
        if (model) {
            const img = tf.browser.fromPixels(imageRef.current);
            const smallImg = tf.image.resizeBilinear(img, [MODEL_SIZE, MODEL_SIZE]);
            const input = tf.sub(tf.div(tf.expandDims(smallImg), 127.5), 1);

            console.time('predict');
            let outputTensor = model.predict(input);
            console.timeEnd('predict');

            const boxes = outputTensor['StatefulPartitionedCall:3'].arraySync()[0];
            const classes = outputTensor['StatefulPartitionedCall:2'].arraySync()[0];
            const probability = outputTensor['StatefulPartitionedCall:1'].arraySync()[0];


            let result = {};
            let resultPoints = {};

            for (let index in probability) {
                if (probability[index] < K) {
                    break;
                }
                const classKey = classes[index];

                if (result[classKey] === undefined) {
                    const [y1, x1, y2, x2] = boxes[index]
                    result[classKey] = {
                        top: y1 * size.height,
                        left: x1 * size.width,
                        right: size.width - (x2 * size.width),
                        bottom: size.height - (y2 * size.height),
                        probability: probability[index].toFixed(2),
                        color: COLOR[classKey]
                    };

                    const leftB = x1 * size.width;
                    const rightB = x2 * size.width;
                    let widthB = rightB - leftB;
                    let centerWidthB = widthB / 2;

                    if (SHIFT[classKey]?.width) {
                        centerWidthB+= widthB * SHIFT[classKey]?.width
                    }
                    const left = leftB + centerWidthB;

                    const topB = y1 * size.height;
                    const bottomB = y2 * size.height;
                    let heightB = bottomB - topB;
                    let centerHeightB = heightB / 2;

                    if (SHIFT[classKey]?.height) {
                        centerHeightB+= heightB * SHIFT[classKey]?.height
                    }
                    const top = topB + centerHeightB;

                    resultPoints[classKey] = {
                        top,
                        left,
                        probability: probability[index].toFixed(2),
                        color: COLOR[classKey]
                    }
                }
            }
            setBox(result);
            setPoints(resultPoints);
        }
    }, [model, size]);

    // eslint-disable-next-line no-unused-vars
    const onLoadImg = useCallback(() => {
        setSize({
            width: wrapRef.current.videoWidth,
            height: wrapRef.current.videoHeight,
        });
        setIsLoadPhoto(true);
    }, [setSize, setIsLoadPhoto]);

    // eslint-disable-next-line no-unused-vars
    const onTakePhoto = useCallback((data) => {
        setIsLoadPhoto(false);
        setPhoto(data);
    }, [setPhoto, setIsLoadPhoto]);

    useEffect(() => {
        if (isLoadPhoto) {
            predict();
        }
    }, [predict, isLoadPhoto])

    console.log(size)
    return (
        <div ref={wrapRef} style={{
            width: '100vw',
            height: '100vh',
            position: 'relative'
        }}>

        <video
          style={{height: '100vh', width: "100vw"}}
          className="size"
          autoPlay
          playsInline
          muted
          ref={videoRef}
          id="frame"
        />
            <div style={{
                position: "absolute",
                border: '3px solid red',
                width: size.width,
                height: size.height,
                top: '50%',
                marginTop: `-${size.height/2}px`
            }}>
                        {Object.entries(points).map(([index, {probability, color, ...style}]) => (
                            <div key={index} style={{
                                position: 'absolute',
                                backgroundColor: color,
                                width: '16px',
                                height: '16px',
                                marginLeft: '-8px',
                                marginTop: '-8px',
                                borderRadius: '8px',
                                color,
                                ...style
                            }}>
                                {/*<div style={{*/}
                                {/*    fontSize: '12px',*/}
                                {/*    position: 'absolute',*/}
                                {/*    top: '-19px',*/}
                                {/*    left: '-3px',*/}
                                {/*    padding: '0px 3px',*/}
                                {/*    backgroundColor: 'white'*/}
                                {/*}}>*/}
                                {/*    {probability}*/}
                                {/*</div>*/}
                            </div>
                        ))}
                {Object.entries(box).map(([index, {probability, color, ...style}]) => (
                    <div key={index} style={{
                        position: 'absolute',
                        border: '2px solid',
                        borderColor: color,
                        color,
                        ...style
                    }}>
                        {/*<div style={{*/}
                        {/*    fontSize: '12px',*/}
                        {/*    position: 'absolute',*/}
                        {/*    top: '-19px',*/}
                        {/*    left: '-3px',*/}
                        {/*    padding: '0px 3px',*/}
                        {/*    backgroundColor: 'white'*/}
                        {/*}}>*/}
                        {/*    {probability}*/}
                        {/*</div>*/}
                    </div>
                ))}
            </div>



            {/*<Camera*/}
            {/*    idealFacingMode = {FACING_MODES.ENVIRONMENT}*/}
            {/*    onTakePhoto={onTakePhoto}*/}
            {/*    isSilentMode={true}*/}
            {/*/>*/}
            {/*{photo && (*/}
            {/*    <div style={{height: `600px`, width: size.width, position: 'relative'}}>*/}
            {/*        /!*<img src={src} alt="" style={{height: `100%`}}/>*!/*/}
            {/*        <img ref={imageRef} src={photo} alt="" style={{height: `100%`}} onLoad={onLoadImg}/>*/}
            {/*        {Object.entries(box).map(([index, {probability, color, ...style}]) => (*/}
            {/*            <div key={index} style={{*/}
            {/*                position: 'absolute',*/}
            {/*                border: '2px solid',*/}
            {/*                borderColor: color,*/}
            {/*                color,*/}
            {/*                ...style*/}
            {/*            }}>*/}
            {/*                /!*<div style={{*!/*/}
            {/*                /!*    fontSize: '12px',*!/*/}
            {/*                /!*    position: 'absolute',*!/*/}
            {/*                /!*    top: '-19px',*!/*/}
            {/*                /!*    left: '-3px',*!/*/}
            {/*                /!*    padding: '0px 3px',*!/*/}
            {/*                /!*    backgroundColor: 'white'*!/*/}
            {/*                /!*}}>*!/*/}
            {/*                /!*    {probability}*!/*/}
            {/*                /!*</div>*!/*/}
            {/*            </div>*/}
            {/*        ))}*/}
            {/*    </div>*/}
            {/*) }*/}
            {/*{model && <p>ready</p>}*/}
        </div>
    );
}

export default App;

import '@tensorflow/tfjs-backend-cpu';
import * as tflite from '@tensorflow/tfjs-tflite';
import * as tf from "@tensorflow/tfjs-core";
import Camera, {FACING_MODES} from 'react-html5-camera-photo';
import 'react-html5-camera-photo/build/css/index.css';
import {useCallback, useEffect, useRef, useState} from "react";
import {config} from "./config";

import {Loading} from './loading'
import {Modal} from './modal'


tflite.setWasmPath('tflite_wasm/');

const MODEL = './model.tflite';
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
const K = 0.5;
const MODEL_SIZE = 320;

// TODO - обавить фото для предзагрузки

function App() {
    const imageRef = useRef();
    const [model, setModel] = useState(null);
    const [photo, setPhoto] = useState(null);
    const [pointIndex, setPointIndex] = useState(null);
    const [points, setPoints] = useState({});

    useEffect(() => {
        tflite.loadTFLiteModel(MODEL).then(setModel)
    }, []);

    const onLoadPhoto = useCallback(() => {
        tf.engine().startScope();
        const size = {
            width: imageRef.current.width,
            height: imageRef.current.height
        };

        const img = tf.browser.fromPixels(imageRef.current);
        const smallImg = tf.image.resizeBilinear(img, [MODEL_SIZE, MODEL_SIZE]);
        const input = tf.sub(tf.div(tf.expandDims(smallImg), 127.5), 1);

        let outputTensor = model.predict(input);

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
                }
            }
        }
        setPoints(resultPoints);

        tf.engine().endScope();

    }, [model]);

    const resetPhoto = useCallback(() => {
        setPhoto(null);
        setPoints({});
    }, [setPhoto, setPoints]);

    const selectPoint = useCallback(function () {
        const {index} = this;
        setPointIndex(index);
    }, [setPointIndex]);

    const onCloseModal = useCallback(() => {
        setPointIndex(null);
    }, [setPointIndex]);

    return (
        <div className={'wrap with-bg'}>
            <div className="wrap-image-hidden">
                {new Array(15).fill('').map((v, i) => <img key={i} src={`images/${i}.png`} alt=""/>)}
            </div>
            {model ? (
                <>
                    <Camera
                        idealFacingMode={FACING_MODES.ENVIRONMENT}
                        onTakePhoto={setPhoto}
                        isImageMirror={false}
                        isFullscreen
                    />

                    <Modal index={pointIndex} onClose={onCloseModal}/>

                    {photo && (
                        <div className={'wrap-photo with-bg'}>
                            <img ref={imageRef} src={photo} alt="" onLoad={onLoadPhoto}/>

                            {Object.entries(points).map(([index, {probability, color, ...style}]) => (
                                config[index] && <div key={index} className={'point'} style={style} onClick={selectPoint.bind({index})} />
                            ))}
                            <button className={'back'} onClick={resetPhoto}>Назад</button>
                        </div>
                    )}
                </>
            ) : <Loading/>}
        </div>
    );
}

export default App;

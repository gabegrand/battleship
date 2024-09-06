import React, { useState } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import BoardComponent from "../components/BoardComponent.jsx";
import ShipsRemainingComponent from "../components/ShipsRemainingComponent.jsx";
import { Button } from "../components/Button.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";
import { noQuestion, noQuestionAnswer, timeoutAnswer } from "../../../utils/systemText.js";
import { SendMessage } from "../../../utils/SendMessage.js";
import { HistoryComponent } from "../components/HistoryComponent.jsx";

export function SharedStage_2() {

    const game = useGame();
    const stage = useStage();
    const timer = useStageTimer();
    const round = useRound();
    const player = usePlayer();
    const [questionRating, setQuestionRating] = useState("");

    function handleQuestionRatingChange(e) {
        setQuestionRating(e.target.value);
    }

    function getAdditionalText(index) {
        switch (index) {
            case 0:
                return "(Not at all helpful)";
            case 1:
                return "(Slightly helpful)";
            case 2:
                return "(Somewhat helpful)";
            case 3:
                return "(Very helpful)";
            case 4:
                return "(Extremely helpful)";
            default:
                return ""
        }
    }

    function handleSpotterSubmission(){
        if (questionRating === "" || questionRating === "(No Rating)") {
            alert("Please select a valid rating before submitting.");
            return;
        }
        round.set("spotterRatings", [...round.get("spotterRatings"), [round.get("question"), parseInt(questionRating, 10)]]);
        stage.set("questionRated",true);
    }
    
    function handleSpotterLikert(){
        if (!round.get("skippedToFiring")) { //&& lastQuestion != undefined
            if (!stage.get("questionRated")) {
                return (
                    <div style={{display:"flex", flexDirection:"row", justifyItems:"center"}}>
                        <div>
                            <label className={"block text-sm font-medium text-gray-700 my-2"}>
                                
                            </label>
                            <div className="flex space-x-4" style={{justifyContent:"center", flexDirection:"row"}}>
                            <p style={{fontSize:"1.25vw", marginRight:"0.5vw"}}>How helpful is this question?</p>
                                <select 
                                    value={questionRating} 
                                    onChange={handleQuestionRatingChange} 
                                    style={{ fontSize: "1vw", padding: "5px" }}
                                >
                                    <option value="(No Rating)">Select a rating...</option>
                                    {[...Array(5)].map((_, index) => (
                                        <option key={index} value={index + 1}>
                                            {index + 1} {getAdditionalText(index)}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>  
                    </div>
                );
            } else {
                return (<div><i>Thank you for rating. The captain is thinking of a question...</i></div>);
            }
        } else {
            player.stage.set("timedOut",false);
            player.stage.set("submit",true);
            return (<div style={{margin: "20px", fontSize:"1vw"}}><i>The captain is thinking of a question...</i></div>);
        }
    }

    function getPossibleAnswers() {
        if (game.get("categoricalAnswers")) {
            return (
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <Button className="m-2" height="3vw" width="8vw" handleClick={() => answerQuestion("categorical", "yes")}>Yes</Button>
                <Button className="m-2" height="3vw" width="8vw" handleClick={() => answerQuestion("categorical", "no")}>No</Button>
            </div>
            );
        } else {
            return (
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <input type="text" id="answer" name="answer" style={{height: "50px"}} autoComplete="off"/>
                <Button className="m-5" handleClick={() => answerQuestion("free")}>Answer Question</Button>
            </div>);
        }
    }

    function answerQuestion(inputType, hardcodedAnswer) {
        let inputText = "";
        if (!round.get("skippedToFiring") && !game.get("timeoutGuiltAssigned")) {
            switch (inputType) {
                case "free":
                    inputText = document.getElementById('answer').value.trim();
                    break;
                case "categorical":
                    inputText = hardcodedAnswer;
                    break;
                default:
                    break;
            }
        }

        if (inputText.length > 1 && inputText.length < 20) {
            handleSpotterSubmission();
            if (questionRating === "" || questionRating === "(No Rating)") {
                return;
            }
            if (!stage.get("answered")){
            SendMessage(inputText, "answer", round, game, timer);
            stage.set("answered", true);
            }
        } 
        if (inputText.length === 0) {
            inputText = round.get("skippedToFiring") ? noQuestionAnswer : "(captain timed out)";
            if (!stage.get("answered")){
                SendMessage(inputText, "answer", round, game, timer);
                stage.set("answered", true);
            }
        }
    }

    if (round.get("question") === undefined) {
        round.set("question", noQuestion);
    }

    switch (player.round.get("role")) {
        case "spotter":
        if (stage.get("answered") || (round.get("skippedToFiring") || game.get("timeoutGuiltAssigned"))){
            player.stage.set("timedOut",false);
            player.stage.set("submit", true);
        }
            if (!game.get("spotterRatesQuestions")) {
                if (!round.get("skippedToFiring") && !game.get("timeoutGuiltAssigned")) {
                    return (
                        <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                                <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                                    <ShipsRemainingComponent shipsStatus={round.get("shipsSunk")} />
                                </div>
                                <SpotterBoardComponent 
                                    occ_tiles={round.get("occTiles")}
                                    init_tiles={round.get("trueTiles")}
                                    ships={round.get("ships")}
                                />
                                <HistoryComponent grid={round.get("occTiles")[0].length} />
                            </div>
                            <p style={{margin: "10px", fontSize:"1.5vw"}}>Captain: <i>"{round.get("question")}"</i>.</p> <p>Please answer below:</p>
                            {getPossibleAnswers()}
                        </div>
                    );
                } else {
                    answerQuestion();
                }
            } else {
                if (!round.get("skippedToFiring") && !game.get("timeoutGuiltAssigned")) {
                    return (
                        <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                                <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                                    <ShipsRemainingComponent shipsStatus={round.get("shipsSunk")} />
                                </div>
                                <SpotterBoardComponent 
                                    occ_tiles={round.get("occTiles")}
                                    init_tiles={round.get("trueTiles")}
                                    ships={round.get("ships")}
                                />
                                <HistoryComponent grid={round.get("occTiles")[0].length} />
                            </div>
                            <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                                <p style={{margin: "10px", fontSize:"1.5vw"}}>Captain: <i>"{round.get("question")}"</i>.</p> 
                                <p>Please answer below:</p>
                                 <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                                    {getPossibleAnswers()}
                                    {handleSpotterLikert()}
                                 </div>
                            </div>
                        </div>
                    );
                } else {
                    answerQuestion();
                }
            }
            break;
        case "captain":
            player.stage.set("timedOut",false);
            player.stage.set("submit", true);
            return (
                <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}> 
                    <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                        <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                            <ShipsRemainingComponent shipsStatus={round.get("shipsSunk")} />
                        </div>
                        <BoardComponent 
                            init_tiles={round.get("occTiles")}
                            ships={round.get("ships")}
                        />
                        <HistoryComponent grid={round.get("occTiles")[0].length} />
                    </div>
                    <p style={{margin: "20px"}}><i>The spotter is answering your question...</i></p>
                </div>
            );
        default:
            return <div>This is the shared stage. You have no role: something's gone wrong!</div>;
    }
}

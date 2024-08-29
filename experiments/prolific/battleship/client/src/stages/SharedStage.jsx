import React, { useState } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import BoardComponent from "../components/BoardComponent.jsx";
import ShipsRemainingComponent from "../components/ShipsRemainingComponent.jsx";
import { Button } from "../components/Button.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";
import { HistoryComponent } from "../components/HistoryComponent.jsx";
import { noQuestion } from "../../../utils/systemText.js";
import { SendMessage } from "../../../utils/SendMessage.js";
import Board, { hasGameEnded } from "./Board.js";
import { NextGameButton } from "../components/NextGameButton.jsx";

export function SharedStage() {

    const stage = useStage();
    const round = useRound();
    const player = usePlayer();
    const game = useGame();
    const timer = useStageTimer();
    const [showTextbox, setShowTextbox] = useState(false);
    const [questionRating, setQuestionRating] = useState(3);

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

    function hasThinkingTimePassed(){
        return timer && timer.remaining < ((game.get("roundDuration")-2)*1000);
    }

    function handleAskQuestion() {
        if (!showTextbox && !game.get("questionEveryTime")) {
            setShowTextbox(true);
        } else {
            askQuestion();
        }
    }

    function isGameOver() {
        var occBoard = new Board(round.get("occTiles")[0].length, round.get("occTiles"), round.get("ships"));
        var trueBoard = new Board(round.get("trueTiles")[0].length, round.get("trueTiles"), round.get("ships"));
        return hasGameEnded(occBoard,trueBoard,round.get("ships")); 
    }

    function handleGameOver() {
        round.set("gameOver",true);
        player.stage.set("submit",true);
    }

    function getGameOverText() {
        return (<div style={{display:"flex", flexDirection:"column", alignItems: "center", marginTop:"20px"}}>
                <p style={{fontSize:"25px"}}>You found all the ships in <b>{round.get("score")}</b> moves! Click the button to continue to the next round.</p>
                <div style={{margin:"10px"}}>
                    <NextGameButton style={{borderRadius:"50%"}} handleClick={() => handleGameOver()}>
                        Next Game
                    </NextGameButton>
                </div>
                </div>);
    }

    function handleCaptainText() {
        if (isGameOver()) {
            return getGameOverText();
        } else {
            if (hasThinkingTimePassed()) {
                if (!game.get("questionEveryTime")) {
                    if (!stage.get("questionAsked")) {
                        return (
                            <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                                <p style={{margin: "20px", fontSize:"1vw"}}>You are the <b>captain</b>. Your goal is to sink all the hidden ships. You can ask the spotter questions to get information about the board and inform where you should shoot next.</p>
                                <div style={{display:"flex", flexDirection:"row", alignItems: "center", fontSize:"1vw"}}>
                                    {showTextbox ? <input type="text" id="question" name="question" style={{height: "50px"}} autocomplete="off"/> : <div></div>}
                                    {round.get("questionsRemaining") != 0 ?
                                    <Button className="m-5" handleClick={handleAskQuestion}>
                                        {showTextbox ? "Submit Question" : "Ask Question"}
                                    </Button> : <div></div>
                                    }
                                    {showTextbox 
                                    ? <div> <Button className="m-5" handleClick={() => setShowTextbox(false)}>Back</Button></div>
                                    : <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>{round.get("questionsRemaining") != 0 ? <p style={{fontSize: "20px"}}>or</p> : <div></div>} <Button className="m-5" handleClick={() => skipToFiring()}>Ready to Fire!</Button></div>}
                                </div>
                                {game.get("categoricalAnswers") ? <p style={{margin: "10px", fontSize: "0.8vw"}}>Remember: the spotter can only answer your questions with "yes" or "no"!</p> : <div></div>}
                                <p style={{margin: "10px", fontSize: "1vw"}}><i>You can ask <b>{round.get("questionsRemaining")}</b> more questions.</i></p>
                            </div>);
                    } else {
                        return (<div><p style={{margin: "20px"}}><i>The spotter is answering your question...</i> </p></div>);
                    }
                    
                } else {
                    if (!stage.get("questionAsked")) {
                        return (
                            <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                                <p style={{margin: "20px", fontSize:"1vw"}}>You are the <b>captain</b>. Your goal is to sink all the hidden ships. You can ask the spotter questions to get information about the board and inform where you should shoot next.</p>
                                <div style={{display:"flex", flexDirection:"row", alignItems: "center", fontSize:"1vw"}}>
                                    <input type="text" id="question" name="question" style={{height: "50px"}} autoComplete="off"/>
                                    <Button className="m-5" handleClick={handleAskQuestion}>
                                        Submit Question
                                    </Button>
                                </div>
                                {game.get("categoricalAnswers") ? <p style={{margin: "10px", fontSize: "0.8vw"}}>Remember: the spotter can only answer your questions with "yes" or "no"!</p> : <div></div>}
                            </div>);
                    } else {
                        return (<div><p style={{margin: "20px"}}><i>The spotter is answering your question...</i> </p></div>); 
                    }
    
                }
            } else {
                return (<div><p style={{margin: "20px"}}><i>Please think about your action for 2.5 seconds...</i> </p></div>); 
            }

            
        }
    }

    function handleSpotterSubmission(){
        round.set("spotterRatings",[...round.get("spotterRatings"), [round.get("question"), questionRating]]);
        stage.set("questionRated",true);
        player.stage.set("submit",true);
    }
    
    function handleSpotterLikert(){
        var lastQuestion = round.get("question");
        console.log("last q", lastQuestion);
            if (lastQuestion != noQuestion && lastQuestion != undefined) {
                if (!stage.get("questionRated")) {
                return (<div style={{display:"flex", flexDirection:"row", justifyItems:"center"}}><div>
                    <label className={"block text-sm font-medium text-gray-700 my-2"}>
                    <p style={{fontSize:"1.25vw", marginRight:"0.5vw"}}>Rate the spotter's last question. <b>How helpful was the question:</b> "{lastQuestion}"?</p>
                    </label>
                    <div className="flex space-x-4" style={{justifyContent:"center"}}>
                      {[...Array(5)].map((_, index) => (
                    <label key={index} className="flex items-center space-x-1">
                      <input
                        type="radio"
                        name="questionRating"
                        value={index + 1}
                        checked={questionRating === (index + 1).toString()}
                        onChange={handleQuestionRatingChange}
                      />
                      {index + 1} {getAdditionalText(index)}
                    </label>
                  ))}
                    </div>
                  </div>
                    <div style={{margin:"1vw"}}>
                    <Button handleClick={handleSpotterSubmission} width="8vw" height="6vh">Rate</Button>
                    </div>    
                </div>);
                }
                else {
                    return (<div><i>Thank you for rating. The captain is thinking of a question...</i></div>);
                }
            } else {
                player.stage.set("submit",true);
                return (<div style={{margin: "20px", fontSize:"1vw"}}><i>The captain is thinking of a question...</i></div>);
            }
    }

    function askQuestion() {
        var inputText = document.getElementById('question').value;
        inputText = inputText.trim();
        if ((inputText.length >= 1 && inputText.length < 100) && round.get("questionsRemaining") > 0) {
            SendMessage(inputText, "question", round, game, timer);
            round.set("questionsRemaining", round.get("questionsRemaining")-1);
            stage.set("questionAsked",true);
            player.stage.set("submit",true);
        }
    }

    function skipToFiring() {
        SendMessage(noQuestion, "question", round, game, timer);
        player.stage.set("submit",true);
    }

    switch (player.round.get("role")) {
        case "spotter":
            player.stage.set("submit", true);
            return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
            
                <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                    <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                        <ShipsRemainingComponent 
                            shipsStatus={round.get("shipsSunk")}
                        />
                    </div>
                    <SpotterBoardComponent 
                        occ_tiles={round.get("occTiles")}
                        init_tiles={round.get("trueTiles")}
                        ships={round.get("ships")}
                    />
                    <HistoryComponent
                        grid = {round.get("occTiles")[0].length}/>
                </div>
                {isGameOver()
                 ? getGameOverText()
                 :  <div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
                    <p style={{margin: "20px", fontSize:"1vw"}}>You are the <b>spotter</b>. Your role is to answer the captain's questions: you can see the complete board, while the captain can only see the blocks that you see as not striped. </p> 
                    <div style={{margin: "20px", fontSize:"1vw"}}><i>The captain is thinking of a question...</i></div>
                    </div>}
                </div>);
        case "captain":
          return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}> 
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                    <ShipsRemainingComponent 
                        shipsStatus={round.get("shipsSunk")}
                    />
                </div>
                <BoardComponent 
                    init_tiles={round.get("occTiles")}
                    ships={round.get("ships")}
                />
                <HistoryComponent
                    grid = {round.get("occTiles")[0].length}/>
            </div>
            {handleCaptainText()}

            </div>); 
        default:
          return <div>This is the shared stage. You have no role: something's gone wrong!</div>;
      }
}

export function Radio({ selected, name, value, label, onChange }) {
    return (
      <label className="text-sm font-medium text-gray-700">
        <input
          className="mr-2 shadow-sm sm:text-sm"
          type="radio"
          name={name}
          value={value}
          checked={selected === value}
          onChange={onChange}
        />
        {label}
      </label>
    );
  }